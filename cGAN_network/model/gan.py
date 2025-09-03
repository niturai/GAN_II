import matplotlib.pyplot as plt
from keras import backend as K
from IPython import display
import tensorflow as tf
import numpy as np
import datetime
import time
import cv2
import os
import csv

from gan_utils.utils import GanUtils, SaltAndPepper


sfun = GanUtils(img_size=128)

class CGAN():
      """
      The function used for generator and discriminator
      """   
      
      def gan_para(self, LAMBDA, OUTPUT_CHANNELS, filtersize, beta, learning_rate, disc_train_iterations, base_path, batch_size):
          """
          The parameters used in definitions
          """
          self.LAMBDA = LAMBDA
          self.OUTPUT_CHANNELS = OUTPUT_CHANNELS
          self.filtersize = filtersize
          self.beta = beta
          self.learning_rate = learning_rate
          self.disc_train_iterations = disc_train_iterations
          self.base_path = base_path
          self.batch_size = batch_size
          
      def Generator(self):
          """
          define the generator with different layers of neuron.
          It returns the generated image according to the downsample and upsample of an input tensor defined by image size.
          """
          
          inputs = tf.keras.layers.Input(shape=[sfun.img_size, sfun.img_size, 1])             # used to instantiate a Keras tensor of a shape given.
    
          down_stack = [
                        sfun.downsample(64, self.filtersize, apply_batchnorm=False),          # downsample the image with different layers and filter
                        sfun.downsample(128, self.filtersize),
                        sfun.downsample(256, self.filtersize),
                        sfun.downsample(512, self.filtersize),
                        sfun.downsample(512, self.filtersize),
                        sfun.downsample(512, self.filtersize),
                        sfun.downsample(512, self.filtersize),
                       ]

          up_stack = [
                      sfun.upsample(512, self.filtersize, apply_dropout=True),                # upsample the image with different layers and filter
                      sfun.upsample(512, self.filtersize, apply_dropout=True),
                      sfun.upsample(512, self.filtersize),
                      sfun.upsample(256, self.filtersize),
                      sfun.upsample(128, self.filtersize),
                      sfun.upsample(64, self.filtersize),
                     ]
        
          initializer = tf.random_normal_initializer(0.0, 0.02)                              # Initializer that generates tensors with a normal distribution, (mean = 0.0, stddev = 0.02 here)
          last = tf.keras.layers.Conv2DTranspose(self.OUTPUT_CHANNELS, self.filtersize,
                                                 strides=2, padding='same',
                                                 kernel_initializer=initializer,
                                                 activation='tanh')                          # spatial deconvolution over images Used to scale up the input shape

          # Downsampling through the model
          x = inputs                                                                         # a Keras tensor of a shape given generated above
          skips = []
          for down in down_stack:
              x = down(x)                                                                    # Give a form of Keras tensor to each neurons in down_stack
              skips.append(x)                                                                # make a layer with each tensor made with neurons
       
          skips = reversed(skips[:-1])                                                       # reverse the arrangement of layer for upsampling

          # Upsampling and establishing the skip connections
          for up, skip in zip(up_stack, skips):
              x = up(x)                                                                     # Give a form of Keras tensor to each neurons in up_stack
              x = tf.keras.layers.Concatenate()([x, skip])                                  # concatenation the upsampling and downsampling

          x = last(x)                                                                       # do the spatial deconvolution over layers
          return tf.keras.Model(inputs = inputs, outputs = x)                               # A model grouping layers into an object with training/inference features to instantiate a Model 
          
          
      def generator_loss(self, disc_generated_output, gen_output, target):
          """
          calculate loss in generator.
          """         
          loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)                # Computes the cross-entropy loss between true labels and predicted labels
                                                                                            # (value in [-inf, inf] when from_logits=True)          
          gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output+0.0) # either 0 or 1

          l1_loss = tf.reduce_mean(tf.abs(target - gen_output))                             # Mean absolute error between target and gen_output

          total_gen_loss = gan_loss + (self.LAMBDA * l1_loss)                               # total generator loss

          return total_gen_loss, gan_loss, l1_loss
         
      def Discriminator(self):
          """
          define the discriminator with different layers of neuron.
          """
          
          initializer = tf.random_normal_initializer(0, 0.02)                              # Initializer that generates tensors with a normal distribution, (mean = 0.0, stddev = 0.02 here)
    
          inp_ = tf.keras.layers.Input(shape=[sfun.img_size, sfun.img_size, 1], name='input signal')  # used to instantiate a Keras tensor of a shape given for input signal
          tar = tf.keras.layers.Input(shape=[sfun.img_size, sfun.img_size, 1], name='target source')  # used to instantiate a Keras tensor of a shape given for target source
    
          inp = tf.keras.layers.Input(shape=(128, 128, 1))
          noisy_inp = SaltAndPepper(prob=self.beta)(inp)    
          x = tf.keras.layers.concatenate([noisy_inp, tar])                                     # take list of tensors, all of the same shape except for the concatenation axis, and returns a single tensor

          down1 = sfun.downsample(32, self.filtersize, False)(x)                          # downsample the input signal and target source with different layers and filtersize
          down2 = sfun.downsample(64, self.filtersize)(down1)                             # downsample down1 with different layers and filtersize
          down3 = sfun.downsample(128, self.filtersize)(down2)                            # downsample down2 with different layers and filtersize
    
          zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)                              # add rows and columns of zeros at the top, bottom, left and right side of the tensor down3
          conv = tf.keras.layers.Conv2D(256, self.filtersize, strides=1,
                                        kernel_initializer=initializer,
                                        use_bias=False)(zero_pad1)                        # spatial convolution over images zero_pad1 and end the downsampling

          batchnorm1 = tf.keras.layers.BatchNormalization()(conv)                         # normalize the mini batches of conv
    
          leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)                            # Leaky version of a Rectified Linear Unit to activate neurons of batchnorm1

          zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)                         # add rows and columns of zeros at the top, bottom, left and right side of leaky_relu image

          last = tf.keras.layers.Conv2D(1, self.filtersize, strides=1, kernel_initializer=initializer)(zero_pad2) # spatial convolution over images zero_pad2

          return tf.keras.Model(inputs=[inp, tar], outputs=last)                          # A model grouping layers into an object (given input signal, target as input and return predicted result)
                    
      def discriminator_loss(self, disc_real_output, disc_generated_output):
          """
          calculate loss in discriminator.
          """
          
          loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)              # Computes the cross-entropy loss between true labels and predicted labels
          real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)       # loss due to Discriminator

          generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output) # loss due to prove that Generator is not producing correct result

          total_disc_loss = real_loss + generated_loss                                    # total loss in Discriminator

          return total_disc_loss
         
      def optional_rotation(self, prediction, groundtruth):
          """
          Check which one, the predicted image or its 180 rotation, is closer to the ground truth.
          Algorithm used : mean Squared Error (MSE)
          Input : predicted image and ground image
          Output : Either predicted image or its 180 rotation depending on mean square error
          """       
           
          prediction = prediction
          prediction_rot = prediction[0, ::-1, ::-1, 0]                                   # 180 degree rotated image
          prediction = prediction[0, :, :, 0]                                             # predicted image
          groundtruth = groundtruth[0, :, :, 0]                                           # ground image
          
          # Calculate Mean Square Error
          diff_pred = cv2.subtract(np.array(prediction), np.array(groundtruth))           # calculate the difference in predicted and ground truth
          mse_pred = np.sum(diff_pred**2)/float(sfun.img_size**2)                         # normalize the difference

          diff_rot = cv2.subtract(np.array(prediction_rot), np.array(groundtruth))        # difference in the rotated prediction and ground truth
          mse_rot = np.sum(diff_rot**2)/float(sfun.img_size**2)                           # normalize this difference as well
             
          if mse_rot < mse_pred:
             return prediction_rot
          else:
             return prediction
            
      def generate_images(self, model, test_input, target, show_diff = False, sampling = False, save_image = False, counter = None):
          """
          Plots four images based on the model: Input, Prediction, Ground truth and (optionally) difference in the fourier plane.  
          Also checks, whether the image or its rotation is closer to the ground truth. 

          If save_images is provided, counter must also be provided. 

          Parameters:
          -----------
          sampling: boolean
                    Whether the sparse sampling (based on a mask) should be applied.
          show_diff: boolean
                     Whether the difference should be displayed. 
          save_images: boolean
                     Whether some images should be saved. 

          Returns:
          --------
          None
          """
          
          prediction = model(test_input, training = True)
          if show_diff:
             plt.figure(figsize=(15, 3))
          else:
             plt.figure(figsize=(15,15))

          prediction = self.optional_rotation(prediction, target)                         # predicted image or its 180 rotated form based on matching nearer with ground truth or target
            
          if show_diff:
              difference = sfun.ff2d_diff(target[0,:,:,0], prediction, sampling=sampling, base_path=self.base_path)                       # difference of target image and predicted image
              difference = 2*((difference - np.min(difference))/(np.max(difference) - np.min(difference))) -1   # Normalize difference which lies in the range [-1, 1]
              display_list = [test_input[0,:,:,0], target[0,:,:,0], prediction, difference]                     # the observed signal, target or ground truth, predicted image and difference of image 
              title = ['Input Image', 'Ground Truth', 'Predicted Image', 'Difference']                          # with title
              N_images = 4
          else:
              display_list = [test_input[0], target[0], prediction]
              title = ['Input Image', 'Ground Truth', 'Predicted Image']
              N_images = 3   
             
          # plot the results
          for i in range(N_images): 
              plt.subplot(1, N_images, i+1)
              plt.title(title[i])
              plt.imshow(display_list[i]*0.5 + 0.5)                                       # Getting the pixel values in the [0, 1] range to plot:
              plt.colorbar(shrink=0.85)
              plt.gca().set_aspect('equal')
              plt.axis('off')
              
          # Save the plots
          if save_image:
             if counter == None:
                raise Warning("You do not have a counter. Images will be overwritten and NOT be saved.")
       
             plot_name = f"testing_image/testing_image/image_{counter}"
        
             if counter == 0:
                if not os.path.exists(f"{plot_name[:-8]}"):
                   os.makedirs(f"{plot_name[:-8]}")
                   print(f"{plot_name[:-8]} created.")
    
             plt.savefig(f"{plot_name}.eps", format='eps', dpi=500)
             plt.savefig(f"{plot_name}.png")
             
                         
    #   def train_step(self, model1, model2, input_image, target, summary_writer=None, step=None):
    #         """
    #         Perform one training step (per replica).
    #         This is called inside the distributed strategy.
    #         """
    #         with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    #             gen_output = model1(input_image, training=True)
    #             disc_real_output = model2([input_image, target], training=True)
    #             disc_generated_output = model2([input_image, gen_output], training=True)

    #             gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(
    #                 disc_generated_output, gen_output, target
    #             )
    #             disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

    #         # Compute gradients
    #         gen_gradients = gen_tape.gradient(gen_total_loss, model1.trainable_variables)
    #         disc_gradients = disc_tape.gradient(disc_loss, model2.trainable_variables)

    #         # Apply gradients
    #         self.gen_optimizer.apply_gradients(zip(gen_gradients, model1.trainable_variables))
    #         self.disc_optimizer.apply_gradients(zip(disc_gradients, model2.trainable_variables))

    #         # Logging
    #         if summary_writer and step is not None:
    #             with summary_writer.as_default():
    #                 tf.summary.scalar("gen_total_loss", gen_total_loss, step=step)
    #                 tf.summary.scalar("gen_gan_loss", gen_gan_loss, step=step)
    #                 tf.summary.scalar("gen_l1_loss", gen_l1_loss, step=step)
    #                 tf.summary.scalar("disc_loss", disc_loss, step=step)

    #         return gen_total_loss, disc_loss


    #   def fit(self, model1_fn, model2_fn, train_ds, test_ds, steps):
    #         """
    #         Training loop with multi-GPU support, logging, and checkpoints.
    #         model1_fn, model2_fn: callables returning new generator/discriminator instances.
    #         """
    #         import datetime, os

    #         strategy = tf.distribute.MirroredStrategy()
    #         print(f"[INFO] Using {strategy.num_replicas_in_sync} GPUs")

    #         GLOBAL_BATCH_SIZE = 16 * strategy.num_replicas_in_sync

    #         # Prepare datasets
    #         train_ds = train_ds.batch(GLOBAL_BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    #         test_ds  = test_ds.batch(GLOBAL_BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


    #         with strategy.scope():
    #             # Build models + optimizers under scope
    #             model1 = model1_fn()
    #             model2 = model2_fn()
    #             self.gen_optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.5)
    #             self.disc_optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.5)

    #             # Checkpoint setup
    #             run_name = (
    #                 f"cGAN_b{self.beta}_disc{self.disc_train_iterations}_"
    #                 f"bs{GLOBAL_BATCH_SIZE}_lr{self.learning_rate}_f{self.filtersize}"
    #             )
    #             checkpoint_dir = os.path.join("models", run_name)
    #             os.makedirs(checkpoint_dir, exist_ok=True)

    #             log_dir = os.path.join("logs", "fit", run_name)
    #             summary_writer = tf.summary.create_file_writer(log_dir)

    #             checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    #             checkpoint = tf.train.Checkpoint(
    #                 generator_optimizer=self.gen_optimizer,
    #                 discriminator_optimizer=self.disc_optimizer,
    #                 generator=model1,
    #                 discriminator=model2,
    #             )

    #         # Wrap train_step in @tf.function and strategy.run
    #         @tf.function
    #         def distributed_train_step(inputs, step):
    #             input_image, target = inputs
    #             per_replica_losses = strategy.run(
    #                 self.train_step, args=(model1, model2, input_image, target, summary_writer, step)
    #             )
    #             return per_replica_losses

    #         # Example image
    #         example_input, example_target = next(iter(test_ds.take(1)))
    #         start = time.time()
    #         counter = 0

    #         for step, inputs in enumerate(train_ds.repeat().take(steps)):
    #             gen_loss, disc_loss = distributed_train_step(inputs, step)

    #             if step % 1000 == 0:
    #                 display.clear_output(wait=True)
    #                 if step != 0:
    #                     print(f"Time for 1000 steps: {time.time() - start:.2f} sec\n")
    #                 start = time.time()

    #                 self.generate_images(
    #                     model1, example_input, example_target,
    #                     show_diff=True, sampling=True, save_image=True, counter=counter
    #                 )
    #                 counter += 1
    #                 print(f"Step: {step // 1000}k")

    #             if (step + 1) % 10 == 0:
    #                 print(".", end="", flush=True)

    #             if (step + 1) % 5000 == 0:
    #                 checkpoint.save(file_prefix=checkpoint_prefix)


      def train_step(self, model1, model2, input_image, target, summary_writer=None, step=None):
            """
            One training step for generator and discriminator.
            """
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                gen_output = model1(input_image, training=True)
                disc_real_output = model2([input_image, target], training=True)
                disc_generated_output = model2([input_image, gen_output], training=True)

                gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(
                    disc_generated_output, gen_output, target
                )
                disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

            # Gradients
            gen_gradients = gen_tape.gradient(gen_total_loss, model1.trainable_variables)
            disc_gradients = disc_tape.gradient(disc_loss, model2.trainable_variables)

            # Apply
            self.gen_optimizer.apply_gradients(zip(gen_gradients, model1.trainable_variables))
            self.disc_optimizer.apply_gradients(zip(disc_gradients, model2.trainable_variables))

            # TensorBoard logging
            if summary_writer and step is not None:
                with summary_writer.as_default():
                    tf.summary.scalar("gen_total_loss", gen_total_loss, step=step)
                    tf.summary.scalar("gen_gan_loss", gen_gan_loss, step=step)
                    tf.summary.scalar("gen_l1_loss", gen_l1_loss, step=step)
                    tf.summary.scalar("disc_loss", disc_loss, step=step)

            return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss


      def fit(self, model1, model2, train_ds, test_ds, steps):
            """
            Training loop for GAN.
            """
            # Init optimizers
            self.gen_optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.5)
            self.disc_optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.5)

            # Logging dirs
            run_name = (
                f"cGAN_b{self.beta}_disc{self.disc_train_iterations}_"
                f"bs{self.batch_size}_lr{self.learning_rate}_f{self.filtersize}"
            )
            checkpoint_dir = os.path.join("models", run_name)
            os.makedirs(checkpoint_dir, exist_ok=True)

            log_dir = os.path.join("logs", "fit", run_name)
            os.makedirs(log_dir, exist_ok=True)
            summary_writer = tf.summary.create_file_writer(log_dir)

            # CSV logger
            csv_path = os.path.join(log_dir, "losses.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "gen_total_loss", "gen_gan_loss", "gen_l1_loss", "disc_loss"])

            # Checkpoint
            checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
            checkpoint = tf.train.Checkpoint(
                generator_optimizer=self.gen_optimizer,
                discriminator_optimizer=self.disc_optimizer,
                generator=model1,
                discriminator=model2,
            )

            # Example data
            example_input, example_target = next(iter(test_ds.take(1)))
            start = time.time()
            counter = 0

            for step, (input_image, target) in enumerate(train_ds.repeat().take(steps)):
                gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = self.train_step(
                    model1, model2, input_image, target, summary_writer, step
                )

                # Write to CSV every 100 steps
                if step % 100 == 0:
                    with open(csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([step, float(gen_total_loss), float(gen_gan_loss),
                                        float(gen_l1_loss), float(disc_loss)])

                if step % 1000 == 0:
                    display.clear_output(wait=True)
                    if step != 0:
                        print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')
                    start = time.time()
                    self.generate_images(model1, example_input, example_target,
                                        show_diff=True, sampling=True, save_image=True, counter=counter)
                    counter += 1
                    print(f"Step: {step//1000}k")

                if (step + 1) % 100 == 0:
                    print('.', end='', flush=True)

                if (step + 1) % 5000 == 0:
                    checkpoint.save(file_prefix=checkpoint_prefix)
