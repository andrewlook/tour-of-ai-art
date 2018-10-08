# -*- coding: utf-8 -*-
"""20181007-revisiting-ai-art-tour.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1L0VpfSrjTthCZVW4qjUEdoXNH0DqFTHC

# What is AI art?

Definition #1: **Making creative use of outputs from an AI *Tool* **

## Example: [pix2pix](https://affinelayer.com/pixsrv/)

Even without much context on how it works, Pix2Pix is fun to play with and yields wierd results.

We'll look at some more in-depth approaches later (time permitting), but for now this is just fun.

![bread_cat](https://media.giphy.com/media/fik7beSODmO75YI6Qd/giphy.gif)

Based on: [*"Image-to-Image Translation with Conditional Adversarial Networks"*, Isola et al, 2017](https://arxiv.org/abs/1611.07004)

<!-- ![pix2pix_pyramid_cat](https://media.giphy.com/media/wHYLEo4Z7kPJwlbO9U/giphy.gif) -->

Definition #2: **Leveraging how AI represents information to deliberately craft an *Effect* **

## Example: "Face Space"

For example, recent generative algorithms have rapidly improved their ability both to learn **latent spaces**, and to **generate** images from any point in these latent spaces.

In fact, those spooky faces earlier came from a visualization of Eigenfaces, which sought to "learn" how to represent faces in a "latent space".

![facespace](https://onionesquereality.files.wordpress.com/2009/02/face_space.jpg)

Source: [*Face Recognition using Eigenfaces,* Turk et al, 1991](http://www.cs.ucsb.edu/~mturk/Papers/mturk-CVPR91.pdf)

## Example: [Toposketch](https://vusd.github.io/toposketch/)

Researchers have built tools to allow visual navigation of latent spaces in order to gain fine-grained control over the generated artifacts.

![toposketch_man_loop](https://media.giphy.com/media/5BYqBxH66X3UobP3SH/giphy.gif)

Definition #3: **Exploring the *Concepts* within AI and what they mean for us**

## Example: Treachery of Imagenet

![img](https://github.com/andrewlook/tour-of-ai-art/raw/master/notebooks/images/tom_white_adversarial_posters.jpg)

[source](https://medium.com/artists-and-machine-intelligence/perception-engines-8a46bc598d57)

## Example: Simulating How Humans Draw

![](https://media.giphy.com/media/OjY1YSX6jFROm02W0J/giphy.gif)

[source](https://deepmind.com/blog/learning-to-generate-images/)

## Framework to discuss AI Art

- Definition #1: **Making creative use of outputs from an AI *Tool* **
- Definition #2: **Leveraging how AI represents information to deliberately craft an *Effect* **
- Definition #3: **Exploring the *Concepts* within AI and what they mean for us**

Moving forwards, we'll use **Tool**, **Effect**, and **Concept** to interpret different examples of art.

# But first...

## What does AI actually mean?

AI is a term used in so many contexts, that it can be confusing for an audience.

Most mentions of AI can be bucketed into one of two broad categories:
- **General AI**: The futuristic aspiration of computers that experience consciousness
- **Specialized AI**: Systems using statistics to learn patterns from data in order to make predictions.

## What kind of AI are we focusing on today?

The **Specialized** kind (also known as **Machine Learning**).

We'll be focused on artistic applications of techniques that have sprung out of real-world research.

Many artistic tools come from people striving to understand algorithms affect our lives.
<!-- related: Explainability / Interpretability, Algorithmic Accountability -->

Sorry to disappoint, but this tutorial won't cover robots that paint:

![not this](https://media.giphy.com/media/8P7fmIpUYBwHgjo322/giphy.gif)

Source: Deepmind [blog](https://deepmind.com/blog/learning-to-generate-images/) [pdf](https://deepmind.com/documents/183/SPIRAL.pdf) [youtube](https://www.youtube.com/watch?v=iSyvwAwa7vk)

## OK, so what is ML?

Machine Learning (usually) refers to a type of algorithm that:
- **learns** patterns and relationships from **training data**
- is built to perform a clearly-defined **task**.

### A Simple Example

Imagine we want to predict housing prices.

We'd probably start with a spreadsheet of prices we know to be true, alongside features for each house that we expect to be related to the price.

![housepricestable](https://cdn-images-1.medium.com/max/1280/1*TE_oNKRRek5io8v_-uZ9PQ.png)

Source: [medium.com/@kabab](https://medium.com/@kabab/linear-regression-with-python-d4e10887ca43)

If we plot price vs. size on this toy dataset, we can start to see a slope

![housepricesscatter](https://cdn-images-1.medium.com/max/1280/1*HUJzcLczeBFRdPPZ5HMcbw.png)

### Predicting price using the house size

If we were to draw a line through these points, the "slope" would be what you multiply the `size` by in order to get the `price` (plus a "bias" term, if the slope doesn't meet the Y-axis right at zero).

The **objective** of an ML algorithm is to find a line to draw through these points that **minimizes the error** in its predictions.

You can visualize the **error** as the distance from each real data point's `Y` coordinate to the **prediction**: the line's value at the corresponding `X` coordinate.

![errorbars](https://cdn-images-1.medium.com/max/1280/1*iBDH0gBBJNs-oLqUT17yng.png)

### What "Learning" Means

The way ML algorithms "learn" is defined by their **objective**.

In this case, they can start with a proposed line to draw, calculate the error, and make a next guess about what the best line could be.

This process gets repeated, and if everything goes according to plan, the errors decrease.

![iterativelearning](https://cdn-images-1.medium.com/max/1280/1*xc5CSmK9d8oeKYxKxenEGg.gif)

## What is art?

Out of respect to artists everywhere, I'm not going to try to define it.

---

What's relevant here is that art can be defined by whoever is creating it.

<img src="https://upload.wikimedia.org/wikipedia/commons/f/f6/Duchamp_Fountaine.jpg" width=400 />
*"Fountain," Marcel Duchamp, 1917* Source: [wikipedia](https://en.wikipedia.org/wiki/Marcel_Duchamp)

# Back to AI Art

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
"""

################################################################################
# Consider using TF DD setup notebook import and running here to demo deepdream.
################################################################################




# # !cp drive/projects/startup_breakfast/3-breakfast-normal-orig.jpg drive/projects/whirlwind
# # !cp drive/projects/startup_breakfast/3-breakfast-normal.jpg drive/projects/whirlwind
# # !cp drive/projects/startup_breakfast/4-breakfast-dream.jpg drive/projects/whirlwind

# breakfast_normal_full = 'drive/projects/whirlwind/3-breakfast-normal-orig.jpg'
# breakfast_normal_cropped = 'drive/projects/whirlwind/3-breakfast-normal.jpg'
# breakfast_dream_cropped = 'drive/projects/whirlwind/4-breakfast-dream.jpg'

# small_img_0 = load_img(breakfast_normal_cropped, dim=224)
# large_img_0 = load_img(breakfast_normal_cropped)

# image_filenames = [
#     breakfast_normal_cropped,
#     breakfast_dream_cropped,
# ]

# # since images can be pretty large (and slow down the notebook to render them),
# # it's handy to have some lower-res versions to quickly experiment with.
# small_imgs = [load_img(f, dim=224) for f in image_filenames]
# # use a helper from lucid to display the images neatly:
# showing.images(small_imgs, labels=image_filenames)

# # also load the full images, in case you want a high-res deepdream
# large_imgs = [load_img(f) for f in image_filenames]
# #showing.images(large_imgs, labels=image_filenames, w=400)

# # for the rest of the notebook to go smoothly, we'll keep a pointer to the first
# # images that we loaded:
# small_img_0 = small_imgs[0]
# large_img_0 = large_imgs[0]

"""### Emulating Surrealism?

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

> "That was totally not my intention, but let's talk about that!"
>
> -- me

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

## How does it work?
"""

# Visualizing the network graph. Be sure expand the "mixed" nodes to see their 
# internal structure. We are going to visualize "Conv2D" nodes.
tmp_def = rename_nodes(graph_def, lambda s:"/".join(s.split('_',1)))
show_graph(tmp_def)

"""## Example: Face Interpolation

When we **interpolate** between points in the space, generating images at each point along the way, it's striking that (almost) every point in between looks like a person.

One can't but help thinking that these algorithms are beginning to learn some fundamental truths about what we as humans all have in common.

![celeb_1hr_man](https://media.giphy.com/media/1Bgck5vkuyNxSFPNmJ/giphy.gif)

Source: [youtube](https://www.youtube.com/watch?v=36lE9tV9vm0)

Paper: [*Progressive Growing of GANs for Improved Quality, Stability, and Variation*, Karras et al, 2018](http://research.nvidia.com/publication/2017-10_Progressive-Growing-of)

# Definition #3: **Exploring the *Concepts* within AI and what they means for us**

Among other things, this could encompass exploring AI's relationships to:
- our society
- the individual
- the nature of consciousness

## My explorations of concepts

## Example: SketchRNN
"""

# TODO sketchrnn / DRAW / paint?

"""## Example: Treachery of Imagenet"""



