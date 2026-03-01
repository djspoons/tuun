# Tuun Documentation

Visit the [Github Pages site](https://djspoons.github.io/tuun/) to view the documentation with the embedded Tuun synthesizer elements.

## Building the Documentation

The `docs` folder has its copy of the Tuun web component. To release changes of that component to the documentation site, re-add the subtree.
```sh
git subtree split --prefix=web -b web-split
git subtree add --prefix=docs/tuun --squash origin web-split
```
