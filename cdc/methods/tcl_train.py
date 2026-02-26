import math
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cdc.utils.misc as misc

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.amp.GradScaler('cuda')

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)

        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                assert parameters is not None
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)

            self._scaler.step(optimizer)

            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)



def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    if norm_type == float('inf'):
        total_norm = max(p.grad.detach().abs().max().cuda() for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cuda() for p in parameters]
            ),
            norm_type,
        )
    return total_norm

def tcl_train_instance(
    model,
    criterion_ins,
    criterion_clu,
    data_loader,
    optimizer,
    epoch,
    loss_scaler,
    args,
    sample_loss_history,      # 新增：记录每个样本的损失历史，形如 {sample_index: [loss1, loss2, ...]}
    derivative_history,       # 新增：记录每个样本损失导数的历史
    current_dataset           # 新增：当前训练数据集（便于剔除样本后重新构造 DataLoader）
):
    model.train(True)
    
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20
    total_ins_loss = 0
    total_clu_loss = 0
    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(data_loader):
        
        x_w = batch['image']
        x_s = batch['image_augmented']
        indices = batch['index'] 

        x_w = x_w.cuda(non_blocking=True)
        x_s = x_s.cuda(non_blocking=True)

        with torch.amp.autocast('cuda'):
            z_i, z_j, c_i, c_j = model(x_w, x_s)
            c_i = F.softmax(c_i, dim=1)
            c_j = F.softmax(c_j, dim=1)
            # 这里建议criterion_ins采用'reduction="none"'以获得每个样本的损失
            loss_ins_all = criterion_ins(torch.concat((z_i, z_j), dim=0))
            loss_clu = criterion_clu(torch.concat((c_i, c_j), dim=0))
            # 计算当前 batch 的平均实例损失
            loss_ins = loss_ins_all.mean()
            loss = loss_ins + loss_clu

        # 记录每个样本的损失值
        # 注意：若z_i, z_j的拼接导致batch size变为原来的2倍，
        # 则需要对应处理 indices，这里假设 loss_ins_all 的前一半对应 x_w，后一半对应 x_s
        batch_size = x_w.size(0)
        loss_ins_values = loss_ins_all.detach().cpu().numpy()  # 得到一个长度为 2*batch_size 的数组
        # 对于每个样本，我们记录两次损失（或可取均值），这里以两次均值为例：
        for i, idx in enumerate(indices):
            loss_val = (loss_ins_values[i] + loss_ins_values[i + batch_size]) / 2.0
            sample_loss_history.setdefault(idx, []).append(loss_val)
            # 若当前记录足够多，则计算导数
            if len(sample_loss_history[idx]) >= args.derivative_order + 1:
                # calculate_derivative为用户自定义函数，计算loss曲线的n阶导数
                latest_derivative = calculate_derivative(sample_loss_history[idx], args.derivative_order)
                derivative_history.setdefault(idx, []).append(latest_derivative)
                # 为防止历史数据无限增长，仅保留最新的 moving_average_rate + k 个
                derivative_history[idx] = derivative_history[idx][- (args.moving_average_rate + args.k):]

        loss_ins_value = loss_ins.item()
        loss_clu_value = loss_clu.item()

        total_ins_loss += loss_ins_value
        total_clu_loss += loss_clu_value

        if not math.isfinite(loss_ins_value) or not math.isfinite(loss_clu_value):
            print("Loss is {}, {}, stopping training".format(loss_ins_value, loss_clu_value))
            sys.exit(1)

        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=True,
        )

        optimizer.zero_grad()
        torch.cuda.synchronize()

        # 定期打印损失信息
        if data_iter_step % print_freq == 0:
            print(f"{header} Step [{data_iter_step}/{len(data_loader)}]: "
                  f"Loss (Instance): {total_ins_loss/(data_iter_step+1):.4f}, "
                  f"Loss (Cluster): {total_clu_loss/(data_iter_step+1):.4f}")

    # 每个 epoch 结束后，根据累计的导数历史判断是否需要剔除部分样本
    if args.threshold > 0:
        excluded_samples = []
        for idx, derivatives in derivative_history.items():
            if len(derivatives) >= args.moving_average_rate + args.k:
                # 使用移动平均对导数进行平滑，correct_moving_average_new_new为用户实现的函数
                ma_derivatives = correct_moving_average_new_new(derivatives, args.moving_average_rate)
                derivative_sum = np.abs(ma_derivatives[-args.k:]).sum()
                if derivative_sum < args.threshold:
                    excluded_samples.append(idx)
        if excluded_samples:
            print(f"Excluding {len(excluded_samples)} samples due to low derivative sum")
            # 根据 excluded_samples 从当前数据集中剔除对应样本
            remaining_indices = [i for i in range(len(current_dataset)) if i not in excluded_samples]
            # 这里假设当前数据集支持传入索引来构造新的数据集
            new_dataset = Subset(current_dataset, remaining_indices)
            # 重新构造 DataLoader（注意：可能需要调整shuffle等参数）
            new_data_loader = DataLoader(new_dataset, batch_size=data_loader.batch_size,
                                         shuffle=True, num_workers=data_loader.num_workers)
            # 返回更新后的data_loader以及统计信息（sample_loss_history、derivative_history 保留以便后续继续跟踪）
            return new_data_loader, total_ins_loss, total_clu_loss
    # 若未剔除样本，则返回原来的data_loader以及当前统计信息
    return data_loader, total_ins_loss, total_clu_loss

def tcl_train(
    model,
    criterion_ins,
    criterion_clu,
    data_loader,
    optimizer,
    epoch,
    loss_scaler,
    args,
):
    model.train(True)
    
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20
    total_ins_loss = 0
    total_clu_loss = 0
    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(data_loader):

        x_w = batch['image']
        x_s = batch['image_augmented']
        x_w = x_w.cuda(non_blocking=True)
        x_s = x_s.cuda(non_blocking=True)

        with torch.amp.autocast('cuda'):
            z_i, z_j, c_i, c_j = model(x_w, x_s)
            c_i = F.softmax(c_i, dim=1)
            c_j = F.softmax(c_j, dim=1)
            loss_ins = criterion_ins(torch.concat((z_i, z_j), dim=0))
            loss_clu = criterion_clu(torch.concat((c_i, c_j), dim=0))
            loss = loss_ins + loss_clu

        loss_ins_value = loss_ins.item()
        loss_clu_value = loss_clu.item()

        total_ins_loss += loss_ins_value
        total_clu_loss += loss_clu_value

        if not math.isfinite(loss_ins_value) or not math.isfinite(loss_clu_value):
            print(
                "Loss is {}, {}, stopping training".format(
                    loss_ins_value, loss_clu_value
                )
            )
            sys.exit(1)

        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=True,
        )

        optimizer.zero_grad()

        torch.cuda.synchronize()

        # Print loss values at regular intervals (e.g., every `print_freq` steps)
        if data_iter_step % print_freq == 0:
            print(f"{header} Step [{data_iter_step}/{len(data_loader)}]: "
                f"Loss (Instance): {total_ins_loss/(data_iter_step+1):.4f}, "
                f"Loss (Cluster): {total_clu_loss/(data_iter_step+1):.4f}")

            
def train_one_epoch(
    model,
    criterion_ins,
    criterion_clu,
    data_loader,
    optimizer,
    epoch,
    loss_scaler,
    args,
):
    model.train(True)
    
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        x_w = batch['image']
        x_s = batch['image_augmented']
        x_w = x_w.cuda(non_blocking=True)
        x_s = x_s.cuda(non_blocking=True)

        with torch.amp.autocast('cuda'):
            z_i, z_j, c_i, c_j = model(x_w, x_s)
            c_i = F.softmax(c_i, dim=1)
            c_j = F.softmax(c_j, dim=1)
            loss_ins = criterion_ins(torch.concat((z_i, z_j), dim=0))
            loss_clu = criterion_clu(torch.concat((c_i, c_j), dim=0))
            loss = loss_ins + loss_clu

        loss_ins_value = loss_ins.item()
        loss_clu_value = loss_clu.item()

        if not math.isfinite(loss_ins_value) or not math.isfinite(loss_clu_value):
            print(
                "Loss is {}, {}, stopping training".format(
                    loss_ins_value, loss_clu_value
                )
            )
            sys.exit(1)

        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=True,
        )
        optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss_ins=loss_ins_value)
        metric_logger.update(loss_clu=loss_clu_value)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def boost_one_epoch(
    model,
    criterion_ins,
    criterion_clu,
    data_loader,
    optimizer,
    epoch,
    loss_scaler,
    pseudo_labels,
    start_epoch,
    cfg,
):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    optimizer.zero_grad()
    
    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        x_w = batch['image']
        x_s = batch['image_augmented']
        x = batch['val']
        index = batch['index']
        x_w = x_w.cuda(non_blocking=True)
        x_s = x_s.cuda(non_blocking=True)
        x = x.cuda(non_blocking=True)
        
        model.eval()
        with torch.amp.autocast('cuda'), torch.no_grad():
            _, _, c = model(x, x, return_ci=False)
            c = F.softmax(c / cfg['criterion']['clu_temp'], dim=1)
            pseudo_labels_cur, index_cur = criterion_ins.generate_pseudo_labels(
                c, pseudo_labels[index].cuda(), index.cuda()
            )
            pseudo_labels[index_cur] = pseudo_labels_cur
            pseudo_index = pseudo_labels != -1
            metric_logger.update(pseudo_num=pseudo_index.sum().item())
            metric_logger.update(
                pseudo_cluster=torch.unique(pseudo_labels[pseudo_index]).shape[0]
            )
        if epoch == start_epoch:
            continue

        model.train(True)
        with torch.amp.autocast('cuda'):
            z_i, z_j, c_j = model(x_w, x_s, return_ci=False)
            #print(z_i.shape, z_j.shape,pseudo_labels[index].shape)
            #print(torch.concat((z_i, z_j), dim=0).shape)
            loss_ins = criterion_ins(
                torch.concat((z_i, z_j), dim=0), pseudo_labels[index].cuda()
            )
            loss_clu = criterion_clu(c_j, pseudo_labels[index].cuda())
            loss = loss_ins + loss_clu

        loss_ins_value = loss_ins.item()
        loss_clu_value = loss_clu.item()

        if not math.isfinite(loss_ins_value) or not math.isfinite(loss_clu_value):
            print(
                "Loss is {}, {}, stopping training".format(
                    loss_ins_value, loss_clu_value
                )
            )
            sys.exit(1)
            
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=True,
        )
        
        optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss_ins=loss_ins_value)
        metric_logger.update(loss_clu=loss_clu_value)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return (
        {k: meter.global_avg for k, meter in metric_logger.meters.items()},
        pseudo_labels,
    )
