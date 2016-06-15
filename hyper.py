import click
from train import train
import numpy as np

@click.command()
@click.option('--run', default=0, help='nb jobs to run', required=False)
@click.option('--where', default=None, help='where to run', required=False)
def hyperjob(run, where):
    from lightjob.cli import load_db
    from lightjob.db import SUCCESS, RUNNING, AVAILABLE, PENDING
    from lightjob.utils import summarize

    def js1():
        jobs = (
            dict(
                lr=lr,
                b1=b1,
                nb_epochs=100,
                model=dict(scale=scale,
                           num_filters_g=num_filters_g,
                           num_filters_d=num_filters_d,
                           start_w=4,
                           start_h=4,
                           filter_size=5,
                           do_batch_norm=True),
                subset_ratio=subset_ratio,
                dataset='mnist',
                w=32,
                h=32,
                c=1,
                model_name='dcgan')
            for scale in np.logspace(-4, -1, 5)
            for num_filters_g in [32, 64, 128]
            for num_filters_d in [4,  8, 16]
            for lr in (0.0002,)
            for b1 in (0.5,)
            for subset_ratio in (0.01, 0.1, 1)
        )
        return list(jobs)

    def js2():
        jobs = (
            dict(
                lr=lr,
                b1=b1,
                nb_epochs=100,
                model=dict(scale=scale,
                           num_filters_g=num_filters_g,
                           num_filters_d=num_filters_d,
                           start_w=4,
                           start_h=4,
                           filter_size=5,
                           do_batch_norm=True),
                subset_ratio=subset_ratio,
                dataset='fonts',
                w=32,
                h=32,
                c=1,
                model_name='dcgan')
            for scale in np.logspace(-4, -1, 5)
            for num_filters_g in [32, 64, 128]
            for num_filters_d in [4,  8, 16]
            for lr in (0.0002,)
            for b1 in (0.5,)
            for subset_ratio in (0.01, 0.1, 1)
        )
        return list(jobs)

    def js3():
        jobs = (
            dict(
                lr=lr,
                b1=b1,
                nb_epochs=200,
                model=dict(scale=scale,
                           num_filters_g=num_filters_g,
                           num_filters_d=num_filters_d,
                           start_w=4,
                           start_h=4,
                           filter_size=5,
                           do_batch_norm=True),
                subset_ratio=subset_ratio,
                dataset='fonts',
                w=32,
                h=32,
                c=1,
                model_name='dcgan')
            for scale in (0.01778279410038923,)
            for num_filters_g in [64, 128, 256, 512, 1024]
            for num_filters_d in [4,  8,  16, 32, 64, 128]
            for lr in (0.0002,)
            for b1 in (0.5,)
            for subset_ratio in (1,)
        )
        return list(jobs)

    def js4():
        jobs = (
            dict(
                lr=lr,
                b1=b1,
                nb_epochs=200,
                model=dict(scale=scale,
                           num_filters_g=num_filters_g,
                           num_filters_d=num_filters_d,
                           start_w=4,
                           start_h=4,
                           filter_size=5,
                           do_batch_norm=True),
                subset_ratio=subset_ratio,
                dataset='fonts',
                w=64,
                h=64,
                c=1,
                model_name='dcgan')
            for scale in (0.01778279410038923,)
            for num_filters_g in [64, 128, 256, 512, 1024]
            for num_filters_d in [4,  8,  16, 32, 64, 128]
            for lr in (0.0002,)
            for b1 in (0.5,)
            for subset_ratio in (1,)
        )
        return list(jobs)

    def js5():
        jobs = (
            dict(
                lr=lr,
                b1=b1,
                nb_epochs=200,
                model=dict(scale=scale,
                           num_filters_g=num_filters_g,
                           num_filters_d=num_filters_d,
                           start_w=4,
                           start_h=4,
                           filter_size=5,
                           do_batch_norm=True),
                subset_ratio=subset_ratio,
                dataset='insects',
                w=64,
                h=64,
                c=3,
                model_name='dcgan')
            for scale in (0.01778279410038923,)
            for num_filters_g in [64, 128, 256, 512, 1024]
            for num_filters_d in [4,  8,  16, 32, 64, 128]
            for lr in (0.0002,)
            for b1 in (0.5,)
            for subset_ratio in (1,)
        )
        return list(jobs)

    def js6():
        jobs = (
            dict(
                seed=42,
                lr=lr,
                b1=b1,
                epoch_start_decay=epoch_start_decay,
                lr_decay=lr_decay,
                l2_coef=l2_coef,
                nb_epochs=200,
                model=dict(scale=scale,
                           num_filters_g=num_filters_g,
                           num_filters_d=num_filters_d,
                           start_w=4,
                           start_h=4,
                           filter_size=5,
                           do_batch_norm=True),
                subset_ratio=subset_ratio,
                dataset='fonts',
                w=64,
                h=64,
                c=1,
                model_name='dcgan')
            for scale in (0.01778279410038923,)
            for num_filters_g in [1024]
            for num_filters_d in [128]
            for lr in (0.0002, 0.001, 0.00001, 0.00005)
            for b1 in (0.5,)
            for epoch_start_decay in (100,)
            for lr_decay in (1, 0.97, 0.99, 0.9)
            for l2_coef in (0, 1e-7, 1e-5, 1e-3)
            for subset_ratio in (1,)
        )
        return list(jobs)


    def insert_jobs():
        nb = 0
        jobs = list(js1())
        nb += insert(js1(), 'js1')
        nb += insert(js2(), 'js2')
        nb += insert(js3(), 'js3')
        nb += insert(js4(), 'js4')
        nb += insert(js5(), 'js5')
        nb += insert(js6(), 'js6')
        return nb

    def insert(jobs, where=''):
        nb = 0
        for content in jobs:
            s = summarize(content)
            outdir = 'results/{}'.format(s)
            nb += db.safe_add_job(content, outdir=outdir, where=where)
        return nb

    def run_jobs(nb=None, where=None):
        kw = {}
        if where is not None:
            kw['where'] = where
        jobs = db.jobs_with(state=AVAILABLE, **kw)
        jobs = list(jobs)
        if nb is not None:
            jobs = jobs[0:nb]
        for j in jobs:
            db.modify_state_of(j['summary'], PENDING)
        print('starting to run')
        for j in jobs:
            db.modify_state_of(j['summary'], RUNNING)
            kw = j['content']
            kw['outdir'] = j['outdir']
            hist = train(**kw)
            db.update({'hist': hist}, j['summary'])
            db.modify_state_of(j['summary'], SUCCESS)

    db = load_db()
    nb = insert_jobs()
    print('nb jobs inserted : {}'.format(nb))
    run_jobs(nb=run, where=where)

