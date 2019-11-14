#!/bin/bash

set -eux

rm -rf _site
bundle exec jekyll serve --incremental --drafts
