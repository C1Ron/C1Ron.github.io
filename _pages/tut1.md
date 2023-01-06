---
layout: page
title:  ""
categories: jekyll update
permalink: /tut1/
---
## Jekyll and Github Pages
This post is written to share how to host a blog using `jekyll` and connect it to Github for deployment via `github-pages`.

The following steps assumes that your working on a Windows machine, have a Github account and has `git-bash` installed on that machine.

### Install Jekyll and Serve from Github
First of all, we need to install `ruby` and the `jekyll` bundler from [ruby-download]. <br/>
Install the `Ruby+Devkit 3.1.2-1` version and keep the default settings. 

When your installation is about to complete, you get prompted by a popup. <br/> 
Check the `Run 'ridk' install` option and click "Finish".

When prompted by the `RubyInstaller`, press 3 to install `MSYS2 and MINGW development toolchain`.

Finally, open the `Start Command Prompt with Ruby` and write
{% highlight python %}$ gem install jekyll bundler {% endhighlight %}

Type
{% highlight python %}$ jekyll -v {% endhighlight %} 
to check if your installation is properly installed.

Then, open `git-bash` and navigate to where you want the parent folder of your local repository to be.

Initialize the `jekyll` site using `git-bash` and change directory to the local repository:
{% highlight python %}
$ jekyll new blogName
$ cd blogName
{% endhighlight %}

Open the file `_config.yml` and set the `baseurl` variable to `\blogName`, where `blogName` is your chosen name of the blog.

Then, initilize the local git repository and create the branch `gh_pages`:
{% highlight python %}
$ git init
$ git checkout -b gh_pages
{% endhighlight %}


The `_config.yml` file on this blog looks like this:
![config]({{site.baseurl}}/images/config.jpg)

When you're ready to test your site, stage and commit your changes:
{% highlight python %}
$ git add .
$ git commit -m "Initial commit"
{% endhighlight %}

My initial blog-folder looks like this in the `Windows-Explorer` window:
![folder]({{site.baseurl}}/images/folder.jpg)

Now you need to initialize an empty remote repository on `github`, with the name `blogName`. <br/>
Make sure that you don't initialize a `readme` file. 

When your repository is generated, then go `Settings > Pages` and find the `Build and deployment` settings.
Make sure they look like this:
![pages]({{site.baseurl}}/images/pages.jpg)

Then, connect the local repository to the remote and push the content:
{% highlight python %}
$ git remote add origin git@github.com:userName/blogName.git
$ git push origin gh_pages
{% endhighlight %}
where `userName` is your Github username.

Wait for about 5 minutes for the site to get deployed on Github. When ready, you'll see it in the `Pages` settings on Github:
![pagesLive]({{site.baseurl}}/images/pagesLive.jpg)

[ruby-download]: https://rubyinstaller.org/downloads/

### Shorter and faster version
If you name your repository as `userName.github.io` and use the `master` branch to host the site, and you have an empty remote repository with the name `userName.github.io`, where `userName` is your username on Github,  then you can get a website up and running:
```
$ jekyll new userName.github.io
$ cd userName.github.io
$ git init
$ git add .
$ git commit -m "init-commit"
$ git remote add origin git@github.com:userName/userName.github.io.git
$ git push origin master
```
Then deploy the site via `github-pages` by selecting the `master` branch in the `Pages` settings of the repository and press `Save`:
![pages_master]({{site.baseurl}}/images/pages_master.jpg) 

### Mathematics
To include mathematics in markdown we need to copy the `_layout` folder from the `minima` theme. One way to do this is to clone the template:
{% highlight python %} $ git clone git@github.com:jekyll/minima.git
{% endhighlight %}
Then cut and paste the `_layout` folder from the cloned repository into your local blog repository. <br/>
Also add the snippet
{% highlight python %}
markdown: kramdown
{% endhighlight %} 
to the `_config.yml` file:
![config_kramdown]({{site.baseurl}}/images/config_kramdown.jpg)

Then, add the following snippet in the `_layout/post.html` file:
{% highlight python %}
{% raw %}
{% if page.mathjax  %}
<script type="text/javascript" async src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=TeX-MML-AM_CHTML'> </script>
{% endif %}
{% endraw %}
{% endhighlight %}
Place the snippet somewhere between `<article>` and `</article>`
![mathjaxSnippet]({{site.baseurl}}/images/mathjaxSnippet.jpg)

To use `mathjax` in a post, add the command 
{% highlight python %}
mathjax: true
{% endhighlight %} 
between the two `---`'s at the start of the file.
![config_mathjax]({{site.baseurl}}/images/config_mathjax.jpg)

Use `\\( )\\` for each inline equation and `$$ $$` for each large equation.

### Figures
To include figures in markdown, you should create an image-folder in your repo.<br/>
The best way to ensure that figures work both locally and remotely is to use the `baseurl` variable. <br/>
I.e. to use relative paths like shown below:
```
{% raw %}![linkTitle]({{site.baseurl}}/images/imageName.jpg){% endraw %}
```