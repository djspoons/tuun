# Tuun Documentation

Visit the [Github Pages site](https://djspoons.github.io/tuun/) to view the documentation with the embedded Tuun synthesizer elements.

## Building the Documentation

The `docs` folder has its copy of the Tuun web component. Originally, the copy was made using the following:
```sh
git subtree split --prefix=web -b web-split
git push -u origin web-split
git subtree add --prefix=docs/tuun --squash origin web-split
```

To release changes of that component to the documentation site, re-add the subtree and then push and pull.
```sh
git subtree split --prefix=web -b web-split
git push -u origin web-split
git subtree pull --prefix=docs/tuun origin web-split --squash
```
