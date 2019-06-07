#!/usr/bin/env python3
# Usage:
#  PYTHONPATH=src ./train --dataset <file|directory|glob>

import argparse
import json
import os
import numpy as np
import tensorflow as tf
import time
import tqdm
from tensorflow.core.protobuf import rewriter_config_pb2

import model, sample, encoder
from load_dataset import load_dataset, Sampler
from accumulate import AccumulatingOptimizer
import memory_saving_gradients

CHECKPOINT_DIR = 'checkpoint'
SAMPLE_DIR = 'samples'


parser = argparse.ArgumentParser(
    description='Fine-tune GPT-2 on your custom dataset.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', metavar='PATH', type=str, required=True, help='Input file, directory, or glob pattern (utf-8 text, or preencoded .npz files).')
parser.add_argument('--model_name', metavar='MODEL', type=str, default='117M', help='Pretrained model name')
parser.add_argument('--combine', metavar='CHARS', type=int, default=50000, help='Concatenate input files with <|endoftext|> separator into chunks of this minimum size')

parser.add_argument('--batch_size', metavar='SIZE', type=int, default=1, help='Batch size')
parser.add_argument('--learning_rate', metavar='LR', type=float, default=0.0001, help='Learning rate for Adam')
parser.add_argument('--accumulate_gradients', metavar='N', type=int, default=1, help='Accumulate gradients across N minibatches.')
parser.add_argument('--memory_saving_gradients', default=False, action='store_true', help='Use gradient checkpointing to reduce vram usage.')
parser.add_argument('--only_train_transformer_layers', default=False, action='store_true', help='Restrict training to the transformer blocks.')

parser.add_argument('--restore_from', type=str, default='latest', help='Either "latest", "fresh", or a path to a checkpoint file')
parser.add_argument('--run_name', type=str, default='run1', help='Run id. Name of subdirectory in checkpoint/ and samples/')
parser.add_argument('--sample_every', metavar='N', type=int, default=100, help='Generate samples every N steps')
parser.add_argument('--sample_length', metavar='TOKENS', type=int, default=1023, help='Sample this many tokens')
parser.add_argument('--sample_num', metavar='N', type=int, default=1, help='Generate this many samples')
parser.add_argument('--save_every', metavar='N', type=int, default=1000, help='Write a checkpoint every N steps')
parser.add_argument('--checkpoint_dir', metavar='PATH', type=str, default='checkpoint', help='Folder Path to write checkpoints')

parser.add_argument('--val_dataset', metavar='PATH', type=str, default=None, help='Dataset for validation loss, defaults to --dataset.')
parser.add_argument('--val_batch_size', metavar='SIZE', type=int, default=2, help='Batch size for validation.')
parser.add_argument('--val_batch_count', metavar='N', type=int, default=40, help='Number of batches for validation.')
parser.add_argument('--val_every', metavar='STEPS', type=int, default=0, help='Calculate validation loss every STEPS steps.')
parser.add_argument('--stop_after', metavar='STOP', type=int, default=None, help='Stop after training counter reaches STOP')


def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass


def main():
    args = parser.parse_args()
    enc = encoder.get_encoder(args.model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', args.model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if args.sample_length > hparams.n_ctx:
        raise ValueError(
            "Can't get samples longer than window size: %s" % hparams.n_ctx)

    if args.model_name == '345M':
        #args.only_train_transformer_layers = True
        args.memory_saving_gradients = True

    CHECKPOINT_DIR=args.checkpoint_dir

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF
    with tf.Session(config=config) as sess:
        context = tf.placeholder(tf.int32, [args.batch_size, None])
        output = model.model(hparams=hparams, X=context)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=context[:, 1:], logits=output['logits'][:, :-1]))

        if args.val_every > 0:
            val_context = tf.placeholder(tf.int32, [args.val_batch_size, None])
            val_output = model.model(hparams=hparams, X=val_context)
            val_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=val_context[:, 1:], logits=val_output['logits'][:, :-1]))
            val_loss_summary = tf.summary.scalar('val_loss', val_loss)


        tf_sample = sample.sample_sequence(
            hparams=hparams,
            length=args.sample_length,
            context=context,
            batch_size=args.batch_size,
            temperature=1.0,
            top_k=40)

        all_vars = [v for v in tf.trainable_variables() if 'model' in v.name]
        train_vars = [v for v in all_vars if '/h' in v.name] if args.only_train_transformer_layers else all_vars
        if args.accumulate_gradients > 1:
            if args.memory_saving_gradients:
                exit("Memory saving gradients are not implemented for gradient accumulation yet.")
            opt = AccumulatingOptimizer(
                opt=tf.train.AdamOptimizer(learning_rate=args.learning_rate),
                var_list=train_vars)
            opt_reset = opt.reset()
            opt_compute = opt.compute_gradients(loss)
            opt_apply = opt.apply_gradients()
            summary_loss = tf.summary.scalar('loss', opt_apply)
        else:
            #opt = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
            opt = AdafactorOptimizer(learning_rate=args.learning_rate, decay_rate=0.8, beta1=0.0, name="Adafactor")

            if args.memory_saving_gradients:
                opt_grads = memory_saving_gradients.gradients(loss, train_vars)
            else:
                opt_grads = tf.gradients(loss, train_vars)
            opt_grads = list(zip(opt_grads, train_vars))
            opt_apply = opt.apply_gradients(opt_grads)
            summary_loss = tf.summary.scalar('loss', loss)

        summary_log = tf.summary.FileWriter(
            os.path.join(CHECKPOINT_DIR, args.run_name))

        saver = tf.train.Saver(
            var_list=all_vars,
            max_to_keep=5,
            keep_checkpoint_every_n_hours=2)
        sess.run(tf.global_variables_initializer())

        if args.restore_from == 'latest':
            ckpt = tf.train.latest_checkpoint(
                os.path.join(CHECKPOINT_DIR, args.run_name))
            if ckpt is None:
                # Get fresh GPT weights if new run.
                ckpt = tf.train.latest_checkpoint(
                    os.path.join('models', args.model_name))
        elif args.restore_from == 'fresh':
            ckpt = tf.train.latest_checkpoint(
                os.path.join('models', args.model_name))
        else:
            ckpt = tf.train.latest_checkpoint(args.restore_from)
        print('Loading checkpoint', ckpt)
        saver.restore(sess, ckpt)

        print('Loading dataset...')
        chunks = load_dataset(enc, args.dataset, args.combine)
        data_sampler = Sampler(chunks)
        if args.val_every > 0:
            val_chunks = load_dataset(enc, args.val_dataset, args.combine) if args.val_dataset else chunks
        print('dataset has', data_sampler.total_size, 'tokens')
        print('Training...')

        if args.val_every > 0:
            # Sample from validation set once with fixed seed to make
            # it deterministic during training as well as across runs.
            val_data_sampler = Sampler(val_chunks, seed=1)
            val_batches = [[val_data_sampler.sample(1024) for _ in range(args.val_batch_size)]
                           for _ in range(args.val_batch_count)]

        counter = 1
        counter_path = os.path.join(CHECKPOINT_DIR, args.run_name, 'counter')
        if os.path.exists(counter_path):
            # Load the step number if we're resuming a run
            # Add 1 so we don't immediately try to save again
            with open(counter_path, 'r') as fp:
                counter = int(fp.read()) + 1

        def save():
            maketree(os.path.join(CHECKPOINT_DIR, args.run_name))
            print(
                'Saving',
                os.path.join(CHECKPOINT_DIR, args.run_name,
                             'model-{}').format(counter))
            saver.save(
                sess,
                os.path.join(CHECKPOINT_DIR, args.run_name, 'model'),
                global_step=counter)
            with open(counter_path, 'w') as fp:
                fp.write(str(counter) + '\n')

        def generate_samples():
            print('Generating samples...')
            context_tokens = data_sampler.sample(1)
            all_text = []
            index = 0
            while index < args.sample_num:
                out = sess.run(
                    tf_sample,
                    feed_dict={context: args.batch_size * [context_tokens]})
                for i in range(min(args.sample_num - index, args.batch_size)):
                    text = enc.decode(out[i])
                    text = '======== SAMPLE {} ========\n{}\n'.format(
                        index + 1, text)
                    all_text.append(text)
                    index += 1
            print(text)
            maketree(os.path.join(SAMPLE_DIR, args.run_name))
            with open(
                    os.path.join(SAMPLE_DIR, args.run_name,
                                 'samples-{}').format(counter), 'w') as fp:
                fp.write('\n'.join(all_text))

        def validation():
            print('Calculating validation loss...')
            losses = []
            for batch in tqdm.tqdm(val_batches):
                losses.append(sess.run(val_loss, feed_dict={val_context: batch}))
            v_val_loss = np.mean(losses)
            v_summary = sess.run(val_loss_summary, feed_dict={val_loss: v_val_loss})
            summary_log.add_summary(v_summary, counter)
            summary_log.flush()
            print(
                '[{counter} | {time:2.2f}] validation loss = {loss:2.2f}'
                .format(
                    counter=counter,
                    time=time.time() - start_time,
                    loss=v_val_loss))

        def sample_batch():
            return [data_sampler.sample(1024) for _ in range(args.batch_size)]


        avg_loss = (0.0, 0.0)
        start_time = time.time()

        try:
            while counter != args.stop_after:
                if counter % args.save_every == 0:
                    save()
                if counter % args.sample_every == 0:
                    generate_samples()
                if args.val_every > 0 and (counter % args.val_every == 0 or counter == 1):
                    validation()

                if args.accumulate_gradients > 1:
                    sess.run(opt_reset)
                    for _ in range(args.accumulate_gradients):
                        sess.run(
                            opt_compute, feed_dict={context: sample_batch()})
                    (v_loss, v_summary) = sess.run((opt_apply, summary_loss))
                else:
                    (_, v_loss, v_summary) = sess.run(
                        (opt_apply, loss, summary_loss),
                        feed_dict={context: sample_batch()})

                summary_log.add_summary(v_summary, counter)

                avg_loss = (avg_loss[0] * 0.99 + v_loss,
                            avg_loss[1] * 0.99 + 1.0)

                print(
                    '[{counter} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f}'
                    .format(
                        counter=counter,
                        time=time.time() - start_time,
                        loss=v_loss,
                        avg=avg_loss[0] / avg_loss[1]))

                counter += 1
        except KeyboardInterrupt:
            print('interrupted')
        finally:
            save()


if __name__ == '__main__':
    main()

   
# Adafactor from tensor2tensor -------------------------------------------------------------

class AdafactorOptimizer(tf.train.Optimizer):
    """Optimizer that implements the Adafactor algorithm.
    Adafactor is described in https://arxiv.org/abs/1804.04235.
    Adafactor is most similar to Adam (Kingma and Ba), the major differences are:
    1. For a two-dimensional AxB weight matrix, Adafactor uses only A+B auxiliary
        parameters to maintain the second-moment estimator, instead of AB.
        This is advantageous on memory-limited systems.  In addition, beta1
        (momentum) is set to zero by default, saving an additional auxiliary
        parameter per weight.  Variables with >=3 dimensions are treated as
        collections of two-dimensional matrices - factorization is over the final
        two dimensions.
    2. Adafactor incorporates "update-clipping" - a scale-invariant analog of
        gradient clipping.  This adds stability
    3. Adafactor does not require an external "learning rate".  By default, it
        incorporates a relative-update-scale schedule, corresponding to
        inverse-square-root learning-rate-decay in ADAM.  We hope this works well
        for most applications.
    ALGORITHM:
    parameter -= absolute_update_scale * clip(grad / grad_scale)
    where:
        absolute_update_scale := relative_update_scale * parameter_scale
        relative_update_scale := min((step_num + 1)**-0.5, 1e-2)
        parameter_scale := max(rms(var)), epsilon2)
        clip(x) := x / max(1.0, rms(x))
        grad_scale := tf.sqrt(v)   (v is the second-moment estimator)
    The second-moment estimator v is maintained in a manner similar to Adam:
    We initialize
    ```
    if var is 2-dimensional:
        v_r <- zeros([num_rows])
        v_c <- zeros([num_cols])
    if var is 0-dimensional or 1-dimensional:
        v <- zeros(shape(var))
    ```
    The update rule is as follows:
    ```
    decay_rate = 1 - (step_num + 1) ^ -0.8
    grad_squared = tf.square(grad) + epsilon1
    if var is 2-dimensional:
        v_r <- decay_rate * v_r + (1 - decay_rate) * reduce_mean(grad_squared, 1)
        v_c <- decay_rate * v_c + (1 - decay_rate) * reduce_mean(grad_squared, 0)
        v = outer_prod(v_r, v_c) / reduce_mean(v_r)
    if var is 0-dimensional or 1-dimensional:
        v <- decay_rate * v + (1 - decay_rate) * grad_squared
    ```
    For variables with >=3 dimensions, we factorize the second-moment accumulator
    over the final 2 dimensions.  See the code for details.
    Several parts of this algorithm are configurable from the initializer.
        multiply_by_parameter_scale:  If True, then compute absolute_update_scale
        as described above.  If False, let absolute_update_scale be the externally
        supplied learning_rate.
        learning_rate: represents relative_update_scale if
        multiply_by_parameter_scale==True, or absolute_update_scale if
        multiply_by_parameter_scale==False.
        decay_rate: Decay rate of the second moment estimator (varies by step_num).
        This should be set to a function such that:
        1-1/(step_num + 1) <= decay_rate(step_num) < 1.0
        beta1: enables momentum, as in Adam.  Uses extra memory if nonzero.
        clipping_threshold: should be >=1.0 or None for no update clipping
        factored: whether to factor the second-moment estimator.  True means
        less memory usage.
    """

    def __init__(self,
                multiply_by_parameter_scale=True,
                learning_rate=None,
                decay_rate=None,
                beta1=0.0,
                clipping_threshold=1.0,
                factored=True,
                use_locking=False,
                name="Adafactor",
                epsilon1=1e-30,
                epsilon2=1e-3):
        """Construct a new Adafactor optimizer.
        See class comment.
        Args:
        multiply_by_parameter_scale: a boolean
        learning_rate: an optional Scalar.
        decay_rate: an optional Scalar.
        beta1: a float value between 0 and 1
        clipping_threshold: an optional float >= 1
        factored: a boolean - whether to use factored second-moment estimator
            for 2d variables
        use_locking: If True use locks for update operations.
        name: Optional name for the operations created when applying gradients.
            Defaults to "AdafactorOptimizer".
        epsilon1: Regularization constant for squared gradient.
        epsilon2: Regularization constant for parameter scale.
        Raises:
        ValueError: if absolute_update_scale and relative_update_scale_fn are both
            present or both absent.
        """
        super(AdafactorOptimizer, self).__init__(use_locking, name)
        self._multiply_by_parameter_scale = multiply_by_parameter_scale
        if learning_rate is None:
            learning_rate = self._learning_rate_default(multiply_by_parameter_scale)
        self._learning_rate = learning_rate
        if decay_rate is None:
            decay_rate = self._decay_rate_default()
        self._decay_rate = decay_rate
        self._beta1 = beta1
        self._clipping_threshold = clipping_threshold
        self._factored = factored
        self._epsilon1 = epsilon1
        self._epsilon2 = epsilon2

    def _should_use_factored_second_moment_estimate(self, shape):
        """Should we use a factored second moment estimator.
        Based on the shape of the variable.
        Args:
        shape: a list of integers
        Returns:
        a boolean
        """
        return self._factored and len(shape) >= 2

    def _create_slots(self, var_list):
        for var in var_list:
            shape = var.get_shape().as_list()
            if self._beta1:
                self._zeros_slot(var, "m", self._name)
            if self._should_use_factored_second_moment_estimate(shape):
                r_val = tf.zeros(shape[:-1], dtype=tf.float32)
                c_val = tf.zeros(shape[:-2] + shape[-1:], dtype=tf.float32)
                self._get_or_make_slot(var, r_val, "vr", self._name)
                self._get_or_make_slot(var, c_val, "vc", self._name)
            else:
                v_val = tf.zeros(shape, dtype=tf.float32)
                self._get_or_make_slot(var, v_val, "v", self._name)

    def _apply_dense(self, grad, var):
        return self._resource_apply_dense(grad, var)

    def _apply_sparse(self, grad, var):
        return self._apply_dense(tf.convert_to_tensor(grad), var)

    def _resource_apply_sparse(self, grad, handle, indices):
        return self._resource_apply_dense(
            tf.convert_to_tensor(tf.IndexedSlices(grad, indices, tf.shape(handle))),
            handle)

    def _parameter_scale(self, var):
        """Estimate the scale of the parameters from the current values.
        We include a minimum value of 0.001 to give it a chance to escape 0
        if it was zero-initialized.
        Instead of using the value, we could impute the scale from the shape,
        as initializers do.
        Args:
        var: a variable or Tensor.
        Returns:
        a Scalar
        """
        return tf.maximum(reduce_rms(var), self._epsilon2)

    def _resource_apply_dense(self, grad, handle):
        var = handle
        grad = tf.to_float(grad)
        grad_squared = tf.square(grad) + self._epsilon1
        grad_squared_mean = tf.reduce_mean(grad_squared)
        decay_rate = self._decay_rate
        update_scale = self._learning_rate
        old_val = var
        if var.dtype.base_dtype == tf.bfloat16:
            old_val = tf.to_float(self._parameter_encoding.decode(old_val))
        if self._multiply_by_parameter_scale:
            update_scale *= tf.to_float(self._parameter_scale(old_val))
        # HACK: Make things dependent on grad.
        # This confounds the XLA rewriter and keeps it from fusing computations
        # across different variables.  This fusion is a bad for HBM usage, since
        # it causes the gradients to persist in memory.
        decay_rate += grad_squared_mean * 1e-30
        update_scale += grad_squared_mean * 1e-30
        # END HACK
        mixing_rate = 1.0 - decay_rate
        shape = var.get_shape().as_list()
        updates = []
        if self._should_use_factored_second_moment_estimate(shape):
            grad_squared_row_mean = tf.reduce_mean(grad_squared, -1)
            grad_squared_col_mean = tf.reduce_mean(grad_squared, -2)
            vr = self.get_slot(var, "vr")
            new_vr = (decay_rate * vr + mixing_rate * grad_squared_row_mean)
            vc = self.get_slot(var, "vc")
            new_vc = (decay_rate * vc + mixing_rate * grad_squared_col_mean)
            vr_update = tf.assign(vr, new_vr, use_locking=self._use_locking)
            vc_update = tf.assign(vc, new_vc, use_locking=self._use_locking)
            updates = [vr_update, vc_update]
            long_term_mean = tf.reduce_mean(new_vr, -1, keepdims=True)
            r_factor = tf.rsqrt(new_vr / long_term_mean)
            c_factor = tf.rsqrt(new_vc)
            x = grad * tf.expand_dims(r_factor, -1) * tf.expand_dims(c_factor, -2)
        else:
            v = self.get_slot(var, "v")
            new_v = decay_rate * v + mixing_rate * grad_squared
            v_update = tf.assign(v, new_v, use_locking=self._use_locking)
            updates = [v_update]
            x = grad * tf.rsqrt(new_v)
        if self._clipping_threshold is not None:
            clipping_denom = tf.maximum(1.0, reduce_rms(x) / self._clipping_threshold)
            x /= clipping_denom
        subtrahend = update_scale * x
        if self._beta1:
            m = self.get_slot(var, "m")
            new_m = self._beta1 * tf.to_float(m) + (1.0 - self._beta1) * subtrahend
            subtrahend = new_m
            new_m = cast_like(new_m, var)
            updates.append(tf.assign(m, new_m, use_locking=self._use_locking))
        new_val = tf.to_float(old_val) - subtrahend
        var_update = tf.assign(var, new_val, use_locking=self._use_locking)
        updates = [var_update] + updates
        return tf.group(*updates)

    def _decay_rate_default(self):
        return adafactor_decay_rate_pow(0.8)

    def _learning_rate_default(self, multiply_by_parameter_scale):
        learning_rate = tf.minimum(tf.rsqrt(step_num() + 1.0), 0.01)
        if not multiply_by_parameter_scale:
            learning_rate *= 0.05
        return learning_rate


def adafactor_decay_rate_adam(beta2):
    t = tf.to_float(tf.train.get_or_create_global_step()) + 1.0
    decay = beta2 * (1.0 - tf.pow(beta2, t - 1.0)) / (1.0 - tf.pow(beta2, t))
    # decay = tf.cond(tf.equal(t, 1.0), lambda: beta2, lambda: decay)
    return decay


def adafactor_decay_rate_pow(exponent):
    return 1.0 - tf.pow((step_num() + 1.0), -exponent)

def step_num():
    return tf.to_float(tf.train.get_or_create_global_step())

def reduce_rms(x):
    return tf.sqrt(tf.reduce_mean(tf.square(x)))

def cast_like(x, y):
    """Cast x to y's dtype, if necessary."""
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)

    if x.dtype.base_dtype == y.dtype.base_dtype:
        return x

    cast_x = tf.cast(x, y.dtype)
    if cast_x.device != x.device:
        x_name = "(eager Tensor)"
        try:
            x_name = x.name
        except AttributeError:
            pass
        tf.logging.warning("Cast for %s may induce copy from '%s' to '%s'", x_name,
                        x.device, cast_x.device)
    return cast_x
