# Tuun Documentation

Visit the [Github Pages site](https://djspoons.github.io/tuun/) to view the documentation with the embedded Tuun synthesizer elements.

## Building the Documentation

First, make sure that the web component is up-to-date and is checked in.
```sh
./build-wasm.sh
git add web/pkg/*
git commit -m "update web component"
```

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
git subtree pull --prefix=docs/tuun --squash origin web-split
```

It's a good idea to check all of examples, including those in the `docs` folder.
```sh
./check-web-examples.sh
```
