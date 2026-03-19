# Tuun Documentation

Visit the [Github Pages site](https://djspoons.github.io/tuun/) to view the documentation with the embedded Tuun synthesizer elements.

The documentation contains its own embedded version of the Tuun web component in the `docs/tuun` directory. That means that if you make changes to Tuun (including the embedded context), you must update the version of the web component in `docs/tuun`.

## Testing the Documentation

It's a good idea to check all of the examples in the `docs` folder. The following extracts any Tuun expressions from the documentation and parses them using a local build of the Tuun native app.
```sh
./check-web-examples.sh docs
```

The documentation site uses Github Pages, but you can test it locally by installing Ruby and then running:
```sh
bundle install
bundle exec jekyll serve
```

Note that this *does* use the embedded copy of the Tuun web component, so either follow the directions below to use `git subtree` or just copy the files over:
``sh
cp -r web/* docs/tuun/
``
As noted below, it's preferable to pull changes using `git subtree` (rather than copying them) when committing changes to the documentation.


## Releasing the Documentation

If you've made any changes to the Tuun web component, make sure that it is up-to-date and is checked in.
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

To release changes of that component to the documentation site, re-add the subtree and then push and pull. Do this before you commit changes to the docs.
```sh
git subtree split --prefix=web -b web-split
git push -u origin web-split
git subtree pull --prefix=docs/tuun --squash origin web-split
```
