from weaver.utils.logger import _logger, warn_n_times
from torch.utils.data import DataLoader
from collections import defaultdict, Counter
from weaver.utils.data.tools import _concat

# Adapted from Weaver w/ demo feature added
def test_load(args):
    """
    Loads the test data.
    :param args:
    :return: test_loaders, data_config
    """
    # keyword-based --data-test: 'a:/path/to/a b:/path/to/b'
    # split --data-test: 'a%10:/path/to/a/*'
    import glob
    file_dict = {}
    split_dict = {}
    for f in args.data_test:
        if ':' in f:
            name, fp = f.split(':')
            if '%' in name:
                name, split = name.split('%')
                split_dict[name] = int(split)
        else:
            name, fp = '', f
        files = glob.glob(fp)
        if name in file_dict:
            file_dict[name] += files
        else:
            file_dict[name] = files

    # sort files
    for name, files in file_dict.items():
        file_dict[name] = sorted(files)

    # apply splitting
    for name, split in split_dict.items():
        files = file_dict.pop(name)
        for i in range((len(files) + split - 1) // split):
            file_dict[f'{name}_{i}'] = files[i * split:(i + 1) * split]
    
    def get_test_loader(name):
        filelist = file_dict[name]

        if args.demo:
            import fnmatch
            prereq = '*_000.root'
            filelist = fnmatch.filter(filelist, prereq)
        
        _logger.info('Running on test file group %s with %d files:\n...%s', name, len(filelist), '\n...'.join(filelist))
        num_workers = min(args.num_workers, len(filelist))
        from weaver.utils.dataset import SimpleIterDataset
        test_data = SimpleIterDataset({name: filelist}, args.data_config, for_training=False,
                                      extra_selection=args.extra_test_selection,
                                      load_range_and_fraction=((0, 1), args.data_fraction),
                                      fetch_by_files=True, fetch_step=1,
                                      name='test_' + name)
        test_loader = DataLoader(test_data, num_workers=num_workers, batch_size=args.batch_size, drop_last=False,
                                 pin_memory=True)
        return test_loader

    import functools
    test_loaders = {name: functools.partial(get_test_loader, name) for name in file_dict}
    data_config = SimpleIterDataset({}, args.data_config, for_training=False).config
    return test_loaders, data_config

def _flatten_label(label, mask=None):
    if label.ndim > 1:
        label = label.view(-1)
        if mask is not None:
            label = label[mask.view(-1)]
    # print('label', label.shape, label)
    return label

def _flatten_preds(model_output, label=None, mask=None, label_axis=1):
    if not isinstance(model_output, tuple):
        # `label` and `mask` are provided as function arguments
        preds = model_output
    else:
        if len(model_output == 2):
            # use `mask` from model_output instead
            # `label` still provided as function argument
            preds, mask = model_output
        elif len(model_output == 3):
            # use `label` and `mask` from model output
            preds, label, mask = model_output

    # preds: (N, num_classes); (N, num_classes, P)
    # label: (N,);             (N, P)
    # mask:  None;             (N, P) / (N, 1, P)
    if preds.ndim > 2:
        preds = preds.transpose(label_axis, -1).contiguous()
        preds = preds.view((-1, preds.shape[-1]))
        if mask is not None:
            preds = preds[mask.view(-1)]
    # print('preds', preds.shape, preds)

    if label is not None:
        label = _flatten_label(label, mask)

    return preds, label, mask

def evaluate_classification(model, test_loader, dev, epoch, for_training=True, loss_func=None, steps_per_epoch=None,
                            eval_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix'],
                            tb_helper=None):
    '''
    Adapted from Weaver with Inputs Return Feature
    '''
    from weaver.utils.nn.metrics import evaluate_metrics
    model.eval()

    data_config = test_loader.dataset.config

    label_counter = Counter()
    total_loss = 0
    num_batches = 0
    total_correct = 0
    entry_count = 0
    count = 0
    scores = []
    labels = defaultdict(list)
    labels_counts = []
    observers = defaultdict(list)
    saved_inputs = defaultdict(list)
    start_time = time.time()
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y, Z in tq:
                # X, y: torch.Tensor; Z: ak.Array
                inputs = [X[k].to(dev) for k in data_config.input_names]
                label = y[data_config.label_names[0]].long().to(dev)
                entry_count += label.shape[0]
                try:
                    mask = y[data_config.label_names[0] + '_mask'].bool().to(dev)
                except KeyError:
                    mask = None
                model_output = model(*inputs)
                logits, label, mask = _flatten_preds(model_output, label=label, mask=mask)
                scores.append(torch.softmax(logits.float(), dim=1).numpy(force=True))

                if mask is not None:
                    mask = mask.cpu()
                for k, v in y.items():
                    labels[k].append(_flatten_label(v, mask).numpy(force=True))
                if not for_training:
                    for k, v in Z.items():
                        observers[k].append(v)

                num_examples = label.shape[0]
                label_counter.update(label.numpy(force=True))
                if not for_training and mask is not None:
                    labels_counts.append(np.squeeze(mask.numpy(force=True).sum(axis=-1)))

                _, preds = logits.max(1)
                loss = 0 if loss_func is None else loss_func(logits, label).item()

                num_batches += 1
                count += num_examples
                correct = (preds == label).sum().item()
                total_loss += loss * num_examples
                total_correct += correct

                tq.set_postfix({
                    'Loss': '%.5f' % loss,
                    'AvgLoss': '%.5f' % (total_loss / count),
                    'Acc': '%.5f' % (correct / num_examples),
                    'AvgAcc': '%.5f' % (total_correct / count)})

                if tb_helper:
                    if tb_helper.custom_fn:
                        with torch.no_grad():
                            tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch,
                                                i_batch=num_batches, mode='eval' if for_training else 'test')

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (entry_count, entry_count / time_diff))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_counter.items())))

    if tb_helper:
        tb_mode = 'eval' if for_training else 'test'
        tb_helper.write_scalars([
            ("Loss/%s (epoch)" % tb_mode, total_loss / count, epoch),
            ("Acc/%s (epoch)" % tb_mode, total_correct / count, epoch),
        ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode=tb_mode)

    scores = np.concatenate(scores)
    labels = {k: _concat(v) for k, v in labels.items()}
    metric_results = evaluate_metrics(labels[data_config.label_names[0]], scores, eval_metrics=eval_metrics)
    _logger.info('Evaluation metrics: \n%s', '\n'.join(
        ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))

    if for_training:
        return total_correct / count
    else:
        # convert 2D labels/scores
        if len(scores) != entry_count:
            if len(labels_counts):
                labels_counts = np.concatenate(labels_counts)
                scores = ak.unflatten(scores, labels_counts)
                for k, v in labels.items():
                    labels[k] = ak.unflatten(v, labels_counts)
            else:
                assert (count % entry_count == 0)
                scores = scores.reshape((entry_count, int(count / entry_count), -1)).transpose((1, 2))
                for k, v in labels.items():
                    labels[k] = v.reshape((entry_count, -1))
        observers = {k: _concat(v) for k, v in observers.items()}
        return total_correct / count, scores, labels, observers