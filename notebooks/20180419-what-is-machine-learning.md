---
ColabURL: https://colab.research.google.com/drive/1L0VpfSrjTthCZVW4qjUEdoXNH0DqFTHC
---

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


