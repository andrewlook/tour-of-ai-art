---
author_email: andrew.m.look@gmail.com
author_name: Andrew Look
colab_url:https://colab.research.google.com/drive/1vKkCidDN6PE1o9BYvRk2Lxx9T-WdFzcx#scrollTo=rH0i1mgF4dIT
dt: '2018-04-19'
project_slug: a_whirlwind_tour_of_ai_art
colab2_url: https://colab.research.google.com/drive/1L0VpfSrjTthCZVW4qjUEdoXNH0DqFTHC
---



# What is AI art?

- Definition #1: **Making creative use of outputs from an AI *Tool* **
- Definition #2: **Leveraging how AI represents information to deliberately craft an *Effect* **
- Definition #3: **Exploring the *Concepts* within AI and what they mean for us**

## Framework to discuss AI Art

Moving forwards, we'll use **Tool**, **Effect**, and **Concept** to interpret different examples of art.



# What does AI actually mean?

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



## What is art?

Out of respect to artists everywhere, I'm not going to try to define it. 
What's relevant here is that art can be defined by whoever is creating it.

<img src="https://upload.wikimedia.org/wikipedia/commons/f/f6/Duchamp_Fountaine.jpg" width=400 />
*"Fountain," Marcel Duchamp, 1917* Source: [wikipedia](https://en.wikipedia.org/wiki/Marcel_Duchamp)

<!-- modern art = never been done before + but you didnt think of it (SLAA) -->



# Definition #1: **Making creative use of outputs from an AI *Tool* **

## Example: [pix2pix](https://affinelayer.com/pixsrv/)

Even without much context on how it works, Pix2Pix is fun to play with and yields wierd results.

We'll look at some more in-depth approaches later (time permitting), but for now this is just fun.

![bread_cat](https://media.giphy.com/media/fik7beSODmO75YI6Qd/giphy.gif)

Based on: [*"Image-to-Image Translation with Conditional Adversarial Networks"*, Isola et al, 2017](https://arxiv.org/abs/1611.07004)

<!-- ![pix2pix_pyramid_cat](https://media.giphy.com/media/wHYLEo4Z7kPJwlbO9U/giphy.gif) -->



# Definition #2: **Leveraging how AI represents information to deliberately craft an *Effect* **

> "I paint not the things I see but the feelings they arouse in me"
>
> -- Franz Kline

## Example: Deepdreams

Can produce a variety of effects - the person running the deepdream may choose
an unexpected result based on the subject feeling created.

<img src="https://github.com/andrewlook/tour-of-ai-art/raw/master/notebooks/images/deepdream_sky1024px.jpeg" width=400 />

<img src="https://github.com/andrewlook/tour-of-ai-art/raw/master/notebooks/images/deepdream_impressionist.jpeg" width=400 />

<img src="https://github.com/andrewlook/tour-of-ai-art/raw/master/notebooks/images/deepdream_surreal.jpeg" width=400 />

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



## Example: "Face Space"

For example, recent generative algorithms have rapidly improved their ability both to learn **latent spaces**, and to **generate** images from any point in these latent spaces.

In fact, those spooky faces earlier came from a visualization of Eigenfaces, which sought to "learn" how to represent faces in a "latent space".

![facespace](https://onionesquereality.files.wordpress.com/2009/02/face_space.jpg)

Source: [*Face Recognition using Eigenfaces,* Turk et al, 1991](http://www.cs.ucsb.edu/~mturk/Papers/mturk-CVPR91.pdf)


## Example: Face Interpolation

When we **interpolate** between points in the space, generating images at each point along the way, it's striking that (almost) every point in between looks like a person.

One can't but help thinking that these algorithms are beginning to learn some fundamental truths about what we as humans all have in common.

![celeb_1hr_man](https://media.giphy.com/media/1Bgck5vkuyNxSFPNmJ/giphy.gif)

Source: [youtube](https://www.youtube.com/watch?v=36lE9tV9vm0)

Paper: [*Progressive Growing of GANs for Improved Quality, Stability, and Variation*, Karras et al, 2018](http://research.nvidia.com/publication/2017-10_Progressive-Growing-of)


## Example: [Toposketch](https://vusd.github.io/toposketch/)

Researchers have built tools to allow visual navigation of latent spaces in order to gain fine-grained control over the generated artifacts.

![toposketch_man_loop](https://media.giphy.com/media/5BYqBxH66X3UobP3SH/giphy.gif)




# Definition #3: **Exploring the *Concepts* within AI and what they means for us**

Among other things, this could encompass exploring AI's relationships to:
- our society
- the individual
- the nature of consciousness

# Definition #3: **Exploring the *Concepts* within AI and what they mean for us**

## Example: Treachery of Imagenet

![img](https://github.com/andrewlook/tour-of-ai-art/raw/master/notebooks/images/tom_white_adversarial_posters.jpg)

[source](https://medium.com/artists-and-machine-intelligence/perception-engines-8a46bc598d57)

## Example: Simulating How Humans Draw

![](https://media.giphy.com/media/OjY1YSX6jFROm02W0J/giphy.gif)

[source](https://deepmind.com/blog/learning-to-generate-images/)

## Example: SketchRNN

### TODO sketchrnn / DRAW / paint?


