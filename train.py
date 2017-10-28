import time
from collections import Iterable

import torch
from torch.autograd import Variable

import utils

def train(batch_manager, model, discr, w_discr, logger, epoch):

    # Initialize meters.
    data_time = utils.AverageMeter()
    net_time = utils.AverageMeter()
    loss_meter_model = utils.AverageMeter()
    loss_meter_discr = utils.AverageMeter()
    eval_meter_model = utils.AverageMeter()
    eval_meter_discr = utils.AverageMeter()

    # Do the job.
    loader = batch_manager.loader # now db shuffled.
    model.model.train()
    discr.model.train()
    t0 = time.time()
    for i, (inputs_model, targets_model, _) in enumerate(loader):

        # Set variables.
        targets_model = targets_model.cuda(async=True)

        # Measure data time.
        data_time.update(time.time() - t0)
        t0 = time.time()

        # Model forward.
        outputs_model, inputs_discr = model.model(Variable(inputs_model))

        # Make discriminator labels.
        targets_discr = outputs_model.data.max(1)[1].eq(targets_model)
        targets_discr = (1 - targets_discr).unsqueeze(1).float()

        # Discriminator forward.
        outputs_discr = discr.model(inputs_discr.detach())

        # Discriminator loss forward.
        loss_discr = discr.criterion(outputs_discr, Variable(targets_discr))
        eval_discr = batch_manager.evaluator_discr(outputs_discr, targets_discr)

        # Discriminator backward.
        discr.optimizer.zero_grad()
        loss_discr.backward()

        # Discriminator update.
        discr.optimizer.step()

        # Model loss forward.
        loss_model = model.criterion(outputs_model, Variable(targets_model)) \
                + w_discr * discr.criterion(discr.model(inputs_discr), Variable(targets_discr.fill_(0)))
        eval_model = batch_manager.evaluator_model(outputs_model, targets_model)

        # Model backward.
        model.optimizer.zero_grad()
        loss_model.backward()

        # Model update.
        model.optimizer.step()

        # Accumulate statistics.
        loss_meter_model.update(loss_model.data[0], targets_model.size(0))
        loss_meter_discr.update(loss_discr.data[0], targets_discr.size(0))
        eval_meter_model.update(eval_model, targets_model.size(0))
        eval_meter_discr.update(eval_discr, targets_discr.size(0))

        # Measure network time.
        net_time.update(time.time() - t0)
        t0 = time.time()

        # Print iteration.
        print('Epoch {0} Batch {1}/{2} '
                'T-data {data_time.val:.2f} ({data_time.avg:.2f}) '
                'T-net {net_time.val:.2f} ({net_time.avg:.2f}) '
                'M-loss {loss_model.val:.2f} ({loss_model.avg:.2f}) '
                'M-eval {eval_model_val} ({eval_model_avg}) '
                'D-loss {loss_discr.val:.2f} ({loss_discr.avg:.2f}) '
                'D-eval {eval_discr_val} ({eval_discr_avg})'.format(
                    epoch, i + 1, len(loader),
                    data_time=data_time,
                    net_time=net_time,
                    loss_model=loss_meter_model,
                    eval_model_val=utils.to_string(eval_meter_model.val),
                    eval_model_avg=utils.to_string(eval_meter_model.avg),
                    loss_discr=loss_meter_discr,
                    eval_discr_val=utils.to_string(eval_meter_discr.val),
                    eval_discr_avg=utils.to_string(eval_meter_discr.avg)))

    # Summerize results.
    perform = eval_meter_model.avg
    if not isinstance(perform, Iterable): perform = [perform]
    logger.write([epoch, loss_meter_model.avg] + perform + [loss_meter_discr.avg, eval_meter_discr.avg])
    print('Summary of training at epoch {epoch:d}.\n'
            '  Number of pairs: {num_sample:d}\n'
            '  Number of batches: {num_batch:d}\n'
            '  Total time for data: {data_time:.2f} sec\n'
            '  Total time for network: {net_time:.2f} sec\n'
            '  Total time: {total_time:.2f} sec\n'
            '  Average model loss: {avg_loss_model:.4f}\n'
            '  Average model performance: {avg_perf_model}\n'
            '  Average discriminator loss: {avg_loss_discr:.4f}\n'
            '  Average discriminator performance: {avg_perf_discr}'.format(
                epoch=epoch,
                num_sample=loss_meter_model.count,
                num_batch=len(loader),
                data_time=data_time.sum,
                net_time=net_time.sum,
                total_time=data_time.sum+net_time.sum,
                avg_loss_model=loss_meter_model.avg,
                avg_perf_model=utils.to_string(eval_meter_model.avg, '%.4f'),
                avg_loss_discr=loss_meter_discr.avg,
                avg_perf_discr=utils.to_string(eval_meter_discr.avg, '%.4f')))
