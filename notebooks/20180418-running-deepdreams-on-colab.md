---
filename: 20181007-running-deepdreams-on-colab.ipynb
colab_url: https://colab.research.google.com/drive/1OKF9JwkeRXQ_pqkPPQIet2teAHOrPBNb
---

# Running Deepdreams on Google Colab

# What to expect

## You Are here because:

- You're curious to learn how AI can augment the creative process

## By the end of this tour

**My goal is to help you understand:**
- How using AI art tools can *augment* your creativity
- Aesthetic controls at a deeper level, for one particular technique *(deepdreams)*

## Following along
you may follow along using this interactive notebook hosted on Google Colab: **[Start Here](https://colab.research.google.com/github/andrewlook/tour-of-ai-art/blob/master/notebooks/a_whirlwind_tour_of_ai_art.ipynb#scrollTo=-88s3MpezIVD)**

TODO(look): update this link

# Start here

All you need to do to follow along are a few steps:
1. Save a copy of the Notebook in your own Google Drive account
2. Change runtime to GPU
3. Upload your photo(s) and display them
4. Kick off the longer setup scripts and wait a few mins
5. Test deepdream just to make sure everything works

## Step 1 - Save a copy

If you want to save your notebook to you own Google Drive account. In the top menu, choose **File -> Save a copy in drive** from the top menu when you're ready to save.

> **Colab Hints**
>
> If you want to learn more about how to navigate Colab:
> - [Tensorflow with GPU](https://colab.research.google.com/notebooks/gpu.ipynb)
> - [Loading External Data](https://colab.research.google.com/notebooks/io.ipynb)

> To get a brief overview of Colab's features:
> - [Hello, Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb#scrollTo=-Rh3-Vt9Nev9)
> - [Overview of Colaboratory Features](https://colab.research.google.com/notebooks/basic_features_overview.ipynb)


## Step 2 - Change Runtime to "GPU"

This makes notebook run much, much faster.

In the top menu, Click **Runtime** and then select **Change Runtime Type**.

Then, click the **Hardware Accelerator** dropdown, select **GPU** and then click **Save**

<img src="https://github.com/andrewlook/tour-of-ai-art/raw/master/notebooks/images/gpu_py2_colab.png" width=400 />

**TODO(look):** fix that image link.

When you're done, running the following code cell should confirm that a GPU is available:

```
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

import sys
print(sys.version)
assert sys.version.startswith('2')
```


# Background on deepdreams

This tool arose from trying to **understand what Neural Networks "see".**

One way to do this is by understanding which area of the image caused the prediction:

<img src="https://github.com/andrewlook/tour-of-ai-art/raw/master/notebooks/images/attribution.png" width=400 />

source: [distill.pub](https://distill.pub/2017/feature-visualization/)

---

Instead, the a-ha moment for the deepdream authors was that we could get the neural network to "hallucinate" and how us *"what picture would make this look more like X"*?

<img src="https://github.com/andrewlook/tour-of-ai-art/raw/master/notebooks/images/featureviz.png" width=400 />


> Pareidolia is a type of apophenia, which is a more generalized term for seeing patterns in random data. Some common examples are seeing a likeness of Jesus in the clouds or an image of a man on the surface of the moon. 
>
> -- [source](https://www.livescience.com/25448-pareidolia.html)

<img src="https://img.purch.com/w/640/aHR0cDovL3d3dy5saXZlc2NpZW5jZS5jb20vaW1hZ2VzL2kvMDAwLzAzNC8zNDMvaTAyL3JvcnNjaGFjaC0wMi5qcGc/MTM1NTI2Njg4MA==" widht=400 />

# Deliberately Crafting an Effect

<img src="https://cdn-images-1.medium.com/max/1600/1*3rECTefgSkJJ6Sni5sxptA.png" width=600 />

## My Work: exploring high and low layers

<img src="https://github.com/andrewlook/tour-of-ai-art/raw/master/notebooks/images/n2.jpg" width=400/>

<img src="https://github.com/andrewlook/tour-of-ai-art/raw/master/notebooks/images/n3.jpg" width=400/>

<img src="https://github.com/andrewlook/tour-of-ai-art/raw/master/notebooks/images/IMG_5907.jpg" width=400 />

<img src="https://github.com/andrewlook/tour-of-ai-art/raw/master/notebooks/images/IMG_7101.JPG" width=400 />

<img src="https://github.com/andrewlook/tour-of-ai-art/raw/master/notebooks/images/naptime_mixed3a_i30.jpeg" width=400 />


"""
## Step 3 - Kick off the longer setup scripts and wait a few mins

There's a bunch of big blobs of code below - if you skip down to a cell below them and run the following, it will execute all cells before that one:
**`<Cmd>-<fn>-F8`**
(Note: you may not need to press `fn` key).

### Jump down here and run all setup cells at once

run the following, it will execute all cells before that one:
**`<Cmd>-<fn>-F8`**
(Note: you may not need to press `fn` key).


```
# First run <Cmd>-<fn>-F8, to run all cells before this one.

# While you're waiting, you can hit <Shift>-Enter in this cell,
# and it will generate some output to verify that everything got set
# up correctly.
```


