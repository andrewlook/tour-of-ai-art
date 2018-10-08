# Back to AI Art

<!-- reference to previous post -->

Definition #1: **Making creative use of outputs from an AI *Tool* **

Definition #2: **Leveraging how AI represents information to deliberately craft an *Effect* **

Definition #3: **Exploring the *Concepts* within AI and what they means for us**

These definitions actually mirror the progression of my own art practice.

I'll step through them in an attempt to share how my understanding of the space has evolved since I first started.



# Definition #1: Making creative use of outputs from an AI *Tool*

## My Work: a first use-case for an AI art tool

I started getting serious about learning ML sometime after starting to paint.

Around the same time, I took an art class about "Painting the Personal Narrative," and we needed to make a past / present series of still life paintings.

I started with a "present" painting, which I called "startup dinner" (a joking reference to my somewhat unhealthy eating habits left over from pre-Pinterest life).

![startup_dinner](https://lh3.googleusercontent.com/GznHwRqvdYWzoju9ND-vi4yXrbGGYPwjqW4gajOcz8Po4RTF1rGI3PrJlCIA4OZQRbw7klE4rBagDRs-HXxpvN5ldqfj4J-sAlziBSwOK_ZcNP5RtZyRwXRP2SWuJgBog1jHH-rmwBRC5ec_lxeT1VNqND21iAVCak547gQffAOa6jGWZC2-6CecRw-thU8pt3c9i-9v8c-unAJ645_ni5CSHQYIsIki5cOBaUBAllbh4pcRGHwnN-wsvmpmIb2sSW_NLSe4RSGnNj163N3vcvU9jMq-NMiNBxVnY4DWJGR4C4mG22Xr_Xp6R6N5m_j-Z3vjU2XMevUEJbuaweP4X8rksBEsAJ_5acPeyL2l8mQuzeYUyfuuPpHE7U8OA3DbWuGBf7UKGg8IheaDB4_QmMnPIsT3rZ49aOGRtIZ_TBGvmqSBnU-0GuzukASTjitd4BVi_v7Oa8B2O7hWZhmflv2u2S-UNtkDE6hWeyZXs3UnMjhjbyy1HicNTEDita-JcwtgENJkfXFKaCJnb9wfMBgSdLiNScKzMXe_xz_z50IwDBZvoyHsj7KHHjsiOcmEy0AKnrqDvmk4OyXsEKe_X1854op0AsrwICw3V6CW-Z98UbgRLg75cP6ss1L7tVKRj1-DDMz-GJOjlxullRygCB0q7BADx0e_NA=w494-h658-no)


```

################################################################################
# Consider using TF DD setup notebook import and running here to demo deepdream.
################################################################################

# !cp drive/projects/startup_breakfast/3-breakfast-normal-orig.jpg drive/projects/whirlwind
# !cp drive/projects/startup_breakfast/3-breakfast-normal.jpg drive/projects/whirlwind
# !cp drive/projects/startup_breakfast/4-breakfast-dream.jpg drive/projects/whirlwind

breakfast_normal_full = 'drive/projects/whirlwind/3-breakfast-normal-orig.jpg'
breakfast_normal_cropped = 'drive/projects/whirlwind/3-breakfast-normal.jpg'
breakfast_dream_cropped = 'drive/projects/whirlwind/4-breakfast-dream.jpg'

small_img_0 = load_img(breakfast_normal_cropped, dim=224)
large_img_0 = load_img(breakfast_normal_cropped)

image_filenames = [
    breakfast_normal_cropped,
    breakfast_dream_cropped,
]

# since images can be pretty large (and slow down the notebook to render them),
# it's handy to have some lower-res versions to quickly experiment with.
small_imgs = [load_img(f, dim=224) for f in image_filenames]
# use a helper from lucid to display the images neatly:
showing.images(small_imgs, labels=image_filenames)

# also load the full images, in case you want a high-res deepdream
large_imgs = [load_img(f) for f in image_filenames]
#showing.images(large_imgs, labels=image_filenames, w=400)

# for the rest of the notebook to go smoothly, we'll keep a pointer to the first
# images that we loaded:
small_img_0 = small_imgs[0]
large_img_0 = large_imgs[0]

```


### Emulating Surrealism?

When we were asked to make a painting of the "past," the prompt was to make it "nightmarish."

I had another photo chosen already, which i jokingly titled "startup breakfast".

The only problem was, I didn't know how to make a dream effect.

<img src="https://raw.githubusercontent.com/andrewlook/tour-of-ai-art/master/notebooks/images/breakfast/breakfast_normal_full.jpeg" width=400 />

### Enter Deepdreams

Thankfully, code for making deepdreams had just been [announced by Google Research](https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html).

![deepdream_examples](https://2.bp.blogspot.com/-nxPKPYA8otk/VYIWRcpjZfI/AAAAAAAAAmE/8dSuxLnSNQ4/s640/image-dream-map.png)

### Running my first Deepdream

<img src="https://github.com/andrewlook/tour-of-ai-art/raw/master/notebooks/images/3-breakfast-normal.jpg" width=400 />

<img src="https://github.com/andrewlook/tour-of-ai-art/raw/master/notebooks/images/4-breakfast-dream.jpg" width=400 />

### Painting it

<img src="https://github.com/andrewlook/tour-of-ai-art/raw/master/notebooks/images/breakfast/paint_2016-11-13-palette-2.jpg" width=400 />

<img src="https://github.com/andrewlook/tour-of-ai-art/raw/master/notebooks/images/breakfast/paint_2016-11-19.jpg" width=400 />

<img src="https://github.com/andrewlook/tour-of-ai-art/raw/master/notebooks/images/breakfast/IMG_5592.jpg" width=400 />


## What's going on here?

> "That looks like a statement about how animals relate to the food we eat"
>
> -- someone in my art class

This tool arose from trying to understand what Neural Networks "see".

One way to do this is by understanding which area of the image caused the prediction:

<img src="https://github.com/andrewlook/tour-of-ai-art/raw/master/notebooks/images/attribution.png" width=400 />

source: [distill.pub](https://distill.pub/2017/feature-visualization/)

Instead, the a-ha moment for the deepdream authors was that we could get the neural network to "hallucinate" and how us *"what picture would make this look more like X"*?

<img src="https://github.com/andrewlook/tour-of-ai-art/raw/master/notebooks/images/featureviz.png" width=400 />

> Pareidolia is a type of apophenia, which is a more generalized term for seeing patterns in random data. Some common examples are seeing a likeness of Jesus in the clouds or an image of a man on the surface of the moon.
>
> -- [source](https://www.livescience.com/25448-pareidolia.html)

<img src="https://img.purch.com/w/640/aHR0cDovL3d3dy5saXZlc2NpZW5jZS5jb20vaW1hZ2VzL2kvMDAwLzAzNC8zNDMvaTAyL3JvcnNjaGFjaC0wMi5qcGc/MTM1NTI2Njg4MA==" widht=400 />

# Definition #2: **Leveraging how AI represents information to deliberately craft an *Effect* **

> "I paint not the things I see but the feelings they arouse in me"
>
> -- Franz Kline


<img src="https://github.com/andrewlook/tour-of-ai-art/raw/master/notebooks/images/deepdream_sky1024px.jpeg" width=400 />

<img src="https://github.com/andrewlook/tour-of-ai-art/raw/master/notebooks/images/deepdream_impressionist.jpeg" width=400 />

<img src="https://github.com/andrewlook/tour-of-ai-art/raw/master/notebooks/images/deepdream_surreal.jpeg" width=400 />

<img src="https://cdn-images-1.medium.com/max/1600/1*3rECTefgSkJJ6Sni5sxptA.png" width=600 />

## My Work: exploring high and low layers

<img src="https://github.com/andrewlook/tour-of-ai-art/raw/master/notebooks/images/n2.jpg" width=400/>

<img src="https://github.com/andrewlook/tour-of-ai-art/raw/master/notebooks/images/n3.jpg" width=400/>

<img src="https://github.com/andrewlook/tour-of-ai-art/raw/master/notebooks/images/IMG_5907.jpg" width=400 />

<img src="https://github.com/andrewlook/tour-of-ai-art/raw/master/notebooks/images/IMG_7101.JPG" width=400 />

<img src="https://github.com/andrewlook/tour-of-ai-art/raw/master/notebooks/images/naptime_mixed3a_i30.jpeg" width=400 />

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



### How do deepdreams work?

Convolutional neural nets have a hierarchical structure:

<img src="https://cdn-images-1.medium.com/max/1600/1*3rECTefgSkJJ6Sni5sxptA.png" width=600 />

Introspecting different computer vision models, motivated users can tune
deepdream settings to achieve or enhance a particular effect.

```
# Visualizing the network graph. Be sure expand the "mixed" nodes to see their
# internal structure. We are going to visualize "Conv2D" nodes.
tmp_def = rename_nodes(graph_def, lambda s:"/".join(s.split('_',1)))
show_graph(tmp_def)
```

# TF RUN CODE

```
mixed4d = 'mixed4d_3x3_bottleneck_pre_relu'
mixed4d_65 = render_lapnorm(T(mixed4d)[:,:,:,65])

# Lower layers produce features of lower complexity.
mixed3b_101 = render_lapnorm(T('mixed3b_1x1_pre_relu')[:,:,:,101])

# optimizing a linear combination of features often gives a "mixture" pattern.
combo__mixed4d_65__mixed3b_101 = render_lapnorm(T(mixed4d)[:,:,:,65]+T(mixed4d)[:,:,:,139], octave_n=4)
```


```
render_lapnorm(T(layer)[:,:,:,65])

# Lower layers produce features of lower complexity.
render_lapnorm(T('mixed3b_1x1_pre_relu')[:,:,:,101])

# optimizing a linear combination of features often gives a "mixture" pattern.
render_lapnorm(T(layer)[:,:,:,65]+T(layer)[:,:,:,139], octave_n=4)
```