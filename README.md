# FC-Convolutional-AutoEncoder
This repo offers sample implementation of a Fully connected and convolutional auto encoder models, trained and analyzed for reconstruction of MNIST digits.

Fully connected autoencoder has an encoder and decoder, with 2 fully connected layers each with latent vector of size 128, whereas the Convoltuional Autoencoder has 3 layers in decoder with an upsampling layer, and latent feature map of size 28,56,40.
<p align='center'>
<img src='https://user-images.githubusercontent.com/53872365/141710079-4a212861-2d10-4367-97c6-b805312b4335.gif'/>
</p>
<h3>QuickStart</h3>

```
conda create -n autoencoder python=3.8 #(optional)
pip install requirements.txt
python3 train_autoenc.py --mode Conv --num_epochs 10 --batch_size 10 --learning_rate 0.1 #Train
python3 test_autoenc.py #For_inference
```
* mod is Conv / Lin depending on you need a Liear encoder.Decoder architecture or Convolutional architecture.
* Creating a conda environment is optional but recomended. 
* Trained model is saved as model.pt in the working directory.
* Hyperparams for training can be changed and the dataset can be altered too.
* You can perform infernce to get results, using test_autoenc.py


<h3>Training/Test Loss Curves for Both Modes</h3>
<p align='center'>
<img src='https://user-images.githubusercontent.com/53872365/141714665-6b181515-faf3-4fb2-b612-000472e54178.png' width="350" height="300" alt>
<img src='https://user-images.githubusercontent.com/53872365/141714738-19b23fdd-8fcb-4bbc-9116-49b86f664946.png' width="350" height="300"alt>
The green line reflects test whereas the blue line reflects the training loss for the respective modes for the autoencoders training over MNIST digit recreation using MSE loss.
</p>

