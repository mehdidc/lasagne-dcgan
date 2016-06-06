@click.command()
@click.option('--outdir', default='', help='Out directory', required=False)
@click.option('--data', default='mnist', help='mnist/fonts', required=False)
@click.option('--cond/--no-cond', default=True, help='Conditional adversarial net', required=False)
def train(outdir, data, cond):
    from data import load_data
    import theano.tensor as T
    from time import time
    from skimage.io import imsave
    #l2 = 2.5e-5
    lr_initial = 0.001
    b1 = 0.5
    nb_epochs = 10000

    z_dim = 100
    batch_size = 128
    nb_discr_updates = 1
    lr = theano.shared(floatX(np.array(lr_initial)))
    rng = np.random.RandomState(12345)

    def preprocess_input(X):
        return X.reshape((X.shape[0], c, w, h))

    def preprocess_output(y):
        return floatX(to_categorical(y))

    # load data
    data_train, data_valid = load_data(data, training_subset=0.01)
    c, w, h = data_train.shape
    nb_outputs = len(set(data_train.y))


    data_train.X = preprocess_input(data_train.X)
    data_train.y = preprocess_output(data_train.y)
    print(data_train.X.shape, data_train.y.shape)

    # display real data
    xdisp = data_train.X[0:100]
    xdisp = xdisp.transpose((0, 2, 3, 1))
    xdisp = xdisp * np.ones((1, 1, 1, 3))
    img = dispims(xdisp, border=1)
    filename = os.path.join(outdir, 'real_data.png')
    imsave(filename, img)

    # compile net
    Y = T.matrix()
    X_real = T.tensor4()
    Z = T.matrix()

    builder_args = dict(z_dim=z_dim, w=w, h=h, c=c)
    if cond:
        builder_args['nb_outputs'] = nb_outputs
        builder = model.cond_dcgan_28x28
    else:
        builder = model.dcgan_28x28

    if cond:
        x_in, y_in, z_in, out_gen, out_discr = builder(**builder_args)
    else:
        x_in, z_in, out_gen, out_discr = builder(**builder_args)

    inputs = {z_in: Z}
    if cond:
        inputs[y_in] = Y
    X_gen = layers.get_output(out_gen, inputs)

    inputs = {x_in: X_real}
    if cond:
        inputs[y_in] = Y
    p_real = layers.get_output(out_discr, inputs)

    inputs = {x_in: X_gen}
    if cond:
        inputs[y_in] = Y
    p_gen = layers.get_output(out_discr, inputs)

    # cost of discr : predict 0 for gen and 1 for real

    d_cost_real = T.nnet.binary_crossentropy(p_real, T.ones(p_real.shape)).mean()
    d_cost_gen = T.nnet.binary_crossentropy(p_gen, T.zeros(p_gen.shape)).mean()

    # cost of gen : make the discr predict 1 for gen
    g_cost_d = T.nnet.binary_crossentropy(p_gen, T.ones(p_gen.shape)).mean()

    d_cost = d_cost_real + d_cost_gen
    g_cost = g_cost_d

    cost = [g_cost, d_cost, g_cost_d, d_cost_real, d_cost_gen]

    discrim_params = layers.get_all_params(out_discr, trainable=True)
    gen_params = layers.get_all_params(out_gen, trainable=True)

    #d_updates = updates.adam(d_cost, discrim_params, learning_rate=lr, beta1=b1)
    #g_updates = updates.adam(g_cost, gen_params, learning_rate=lr, beta1=b1)

    d_updates = updates.rmsprop(d_cost, discrim_params, learning_rate=lr)
    g_updates = updates.rmsprop(g_cost, gen_params, learning_rate=lr)

    all_updates = d_updates.copy()
    all_updates.update(g_updates)

    inputs = [X_real, Z]
    if cond:
        inputs.append(Y)
    train_g = theano.function(inputs, cost, updates=g_updates)
    train_d = theano.function(inputs, cost, updates=d_updates)

    inputs = [Z]
    if cond:
        inputs.append(Y)
    gen = theano.function(inputs, X_gen)

    # Train
    model_filename = os.path.join(outdir, 'model.pkl')
    save_model(builder, builder_args,
               out_gen, out_discr,
               model_filename)
    mkdir_path(outdir)

    history = []
    n_updates = 0
    for epoch in range(1, nb_epochs + 1):
        total_g_loss = 0
        total_d_loss = 0
        nb_g_updates = 0
        nb_d_updates = 0
        t = time()
        for train_X, train_y in tqdm(iterate_minibatches(data_train.X, data_train.y, batch_size)):
            train_Z = floatX(rng.uniform(-1., 1., size=(len(train_X), z_dim)))

            inputs = [train_X, train_Z]
            if cond:
                inputs.append(train_y)
            if n_updates % (nb_discr_updates + 1) == 0:
                total_g_loss += (train_g(*inputs))[0]
                nb_g_updates += 1
            else:
                total_d_loss += (train_d(*inputs))[1]
                nb_d_updates += 1
            n_updates += 1
        stats = OrderedDict()
        stats['epoch'] = epoch
        if nb_g_updates > 0:
            stats['g_loss'] = total_g_loss / nb_g_updates
        if nb_d_updates > 0:
            stats['d_loss'] = total_d_loss / nb_d_updates
        stats['train_time'] = time() - t
        history.append(stats)

        print(tabulate([stats], 'keys'))

        if epoch % 5 == 0:
            if cond:
                nb_samples_per_output = 40
                nb_samples = nb_outputs * nb_samples_per_output
            else:
                nb_samples_per_output = 40
                nb_samples = nb_samples_per_output * 10
            sample_Z = floatX(rng.uniform(-1., 1., size=(nb_samples, z_dim)))
            if cond:
                sample_Y = floatX(to_categorical(np.repeat(np.arange(0, nb_outputs), nb_samples_per_output)))
                sample_X = gen(sample_Z, sample_Y)
            else:
                sample_X = gen(sample_Z)
            img = dispims(sample_X.reshape((sample_X.shape[0], w, h, 1)), border=1)
            filename = os.path.join(outdir, 'samples{:05d}.png'.format(epoch))
            imsave(filename, img)
            save_model(builder, builder_args, out_gen, out_discr, model_filename)

