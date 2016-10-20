import click
import numpy as np
import os

def js1(rng):
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
    return rng.choice(list(jobs))

def js2(rng):
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
    return rng.choice(list(jobs))

def js3(rng):
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
    return rng.choice(list(jobs))

def js4(rng):
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
    return rng.choice(list(jobs))

def js5(rng):
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
    return rng.choice(list(jobs))

def js6(rng):
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
    return rng.choice(list(jobs))

def js7(rng):
    def gen():
        return dict(
                    seed=42,
                    lr=rng.choice((0.0002, 0.001, 0.00001, 0.00005,0.000001)),
                    b1=0.5,
                    epoch_start_decay=50,
                    lr_decay=rng.choice((1, 0.97, 0.99, 0.9)),
                    l2_coef=rng.choice((0, 1e-7, 1e-5, 1e-3)),
                    nb_epochs=200,
                    model=dict(scale=rng.choice(np.logspace(-4, -1, 5)),
                               patch_size=rng.choice((2, 3)),
                               n_steps=rng.randint(5, 40),
                               n_units=rng.randint(1, 5) * 100,
                               n_layers=rng.choice((1, 2, 3))
                              ),
                    subset_ratio=1,
                    dataset='mnist',
                    w=28,
                    h=28,
                    c=1,
                    model_name='brush')
    return gen()

def js8(rng):
    def gen():
        return dict(
                    seed=42,
                    lr=rng.choice((0.0002, 0.001, 0.00001, 0.00005,0.000001)),
                    b1=0.5,
                    epoch_start_decay=50,
                    lr_decay=rng.choice((1, 0.97, 0.99, 0.9)),
                    l2_coef=rng.choice((0, 1e-7, 1e-5, 1e-3)),
                    nb_epochs=200,
                    model=dict(scale=rng.choice(np.logspace(-4, -1, 5)),
                               patch_size=rng.choice((2, 3)),
                               n_steps=rng.randint(5, 40),
                               n_units=rng.randint(1, 5) * 100,
                               n_layers=rng.choice((1, 2, 3))
                              ),
                    subset_ratio=1,
                    dataset='mnist',
                    w=16,
                    h=16,
                    c=1,
                    model_name='brush')
    return gen()

def js9(rng):
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
            subset_ratio=1,
            pattern=os.getenv('DATA_PATH') + '/celeba/**/*.png',
            nb_examples=100000,
            discriminator_loss='feature_matching',
            w=64,
            h=64, 
            c=3,
            model_name='dcgan')
        for scale in np.logspace(-4, -1, 5)
        for num_filters_g in [32, 64, 128]
        for num_filters_d in [4,  8, 16]
        for lr in (0.0002,)
        for b1 in (0.5,)
    )
    return rng.choice(list(jobs))

@click.command()
@click.option('--run', default=0, help='nb jobs to run', required=False)
@click.option('--where', default=None, help='where to run', required=False)
@click.option('--job_id', default=None, help='list of ids separated by ,', required=False)
@click.option('--dry/--no-dry', default=False, help='if dry is True dont modify the state of db', required=False)
def runhyper(run, where, job_id, dry):
    from train import train
    from lightjob.cli import load_db
    from lightjob.db import SUCCESS, RUNNING, AVAILABLE, PENDING
    from lightjob.utils import summarize

    def run_jobs(nb=None, where=None, job_id=None):
        kw = {}
        if where is not None:
            kw['where'] = where
        jobs = db.jobs_with(state=AVAILABLE, **kw)
        jobs = list(jobs)
        if job_id is not None:
            job_id = set(job_id.split(','))
            jobs = [j for j in jobs if j['summary'] in job_id]
        if nb is not None:
            jobs = jobs[0:nb]
        for j in jobs:
            if not dry:
                db.modify_state_of(j['summary'], PENDING)
        print('starting to run')
        for j in jobs:
            params = j['content']
            params['outdir'] = j['outdir']
            db.modify_state_of(j['summary'], RUNNING)
            hist = train(params)
            if not dry:
                db.update({'hist': hist}, j['summary'])
                db.modify_state_of(j['summary'], SUCCESS)
    db = load_db()
    run_jobs(nb=run, where=where, job_id=job_id)

@click.command()
@click.option('--where', default=None, help='where', required=False)
@click.option('--nb', default=1, help='nb', required=False)
def inserthyper(where, nb):
    from lightjob.cli import load_db
    from lightjob.db import SUCCESS, RUNNING, AVAILABLE, PENDING
    from lightjob.utils import summarize
    rng = np.random
    js = globals()[where]
    jobs = [js(rng) for _ in range(nb)]
    nb = 0
    db = load_db()
    for content in jobs:
        s = summarize(content)
        print(s)
        outdir = 'results/{}'.format(s)
        nb += db.safe_add_job(content, outdir=outdir, where=where)
    print('{} of jobs inserted'.format(nb))
    db.close()
