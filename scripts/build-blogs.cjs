const fs = require("fs");
const path = require("path");
const { marked } = require("marked");
const katex = require("katex");
const root = path.resolve(__dirname, "..");
const site = "https://joesun-421.github.io";
const posts = JSON.parse(fs.readFileSync(path.join(root, "blogs/posts.json"), "utf8"));
const escape = value => String(value).replace(/[&<>"']/g, c =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

function frame(title, description, canonical, body, math, structured) {
  return [
    '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">',
    '<meta name="viewport" content="width=device-width,initial-scale=1">',
    '<meta name="theme-color" content="#f3f2ed">',
    '<meta name="description" content="' + escape(description) + '">',
    '<meta property="og:type" content="' + (math ? "article" : "website") + '">',
    '<meta property="og:title" content="' + escape(title) + '">',
    '<meta property="og:description" content="' + escape(description) + '">',
    '<meta property="og:url" content="' + canonical + '">',
    '<title>' + escape(title) + '</title><link rel="canonical" href="' + canonical + '">',
    '<link rel="stylesheet" href="/blogs/blog.css">',
    math ? '<link rel="stylesheet" href="/assets/vendor/katex-0.19.1/katex.min.css">' : "",
    structured ? '<script type="application/ld+json">' +
      JSON.stringify(structured).replace(/</g, "\\u003c") + "</script>" : "",
    '</head><body><div class="wrap">',
    '<header class="topbar"><a class="brand" href="/" aria-label="Zhaoyuan Sun, home">ZS.</a>',
    '<nav class="nav" aria-label="Main navigation"><a href="/">Home</a><a href="/blogs/">Blogs</a><a href="/ideas.html">Ideas</a></nav></header>',
    body,
    '<footer class="site-footer"><span>© 2026 Zhaoyuan Sun</span><a href="/blogs/">Blogs</a><a href="/">Home</a></footer>',
    "</div></body></html>"
  ].join("\n");
}

function compileMarkdown(source) {
  let text = source.replace(/^---\n[\s\S]*?\n---\n/, "").trimStart().replace(/^# [^\n]+\n/, "");
  const formulae = [];
  function protect(tex, display) {
    const i = formulae.length;
    const html = katex.renderToString(tex, {
      displayMode: display, throwOnError: true, strict: "error",
      output: "htmlAndMathml", trust: false
    });
    formulae.push({ display, html });
    return "BLOGMATHPLACEHOLDER" + i + "END";
  }
  text = text.replace(/^\$\$\s*\n([\s\S]*?)\n\$\$$/gm,
    (_, tex) => "\n\n" + protect(tex.trim(), true) + "\n\n");
  text = text.replace(/(?<!\\)\$([^\$\n]+?)(?<!\\)\$/g,
    (_, tex) => protect(tex, false));
  let html = marked.parse(text, { gfm: true });
  formulae.forEach((f, i) => {
    const key = "BLOGMATHPLACEHOLDER" + i + "END";
    html = f.display
      ? html.replace("<p>" + key + "</p>", '<div class="math-block">' + f.html + "</div>")
      : html.replaceAll(key, f.html);
  });
  if (html.includes("BLOGMATHPLACEHOLDER")) throw new Error("Unrendered formula");
  const headings = [], seen = new Map();
  html = html.replace(/<h([23])>([\s\S]*?)<\/h\1>/g, (_, level, label) => {
    const plain = label.replace(/<[^>]+>/g, "");
    const base = plain.toLowerCase().replace(/[^\p{L}\p{N} -]/gu, "").trim().replace(/\s+/g, "-");
    const n = seen.get(base) || 0;
    seen.set(base, n + 1);
    const id = base + (n ? "-" + n : "");
    if (level === "2") headings.push({ id, label: plain });
    return "<h" + level + ' id="' + id + '">' + label + "</h" + level + ">";
  });
  return { html, headings, math: formulae.length, displayMath: formulae.filter(f => f.display).length };
}

const reports = [];
for (const post of posts) {
  const dir = path.join(root, "blogs", post.slug);
  const source = fs.readFileSync(path.join(dir, "source.md"), "utf8");
  const compiled = compileMarkdown(source);
  const canonical = site + "/blogs/" + post.slug + "/";
  const toc = '<details class="toc"><summary>文章目录</summary><ol>' +
    compiled.headings.map(h => '<li><a href="#' + h.id + '">' + escape(h.label) + "</a></li>").join("") +
    "</ol></details>";
  const body = [
    '<main><header class="post-header"><div class="eyebrow">Research notes · Dynamic reasoning</div>',
    "<h1>" + escape(post.title) + "</h1>",
    '<div class="post-meta"><time datetime="' + post.date + '">' + post.date + "</time>",
    "<span>Zhaoyuan Sun</span>",
    '<a href="https://github.com/JoeSun-421/joesun-421.github.io/blob/main/blogs/' + post.slug + '/source.md">Markdown source</a></div></header>',
    toc, '<article class="prose">' + compiled.html + "</article></main>"
  ].join("\n");
  fs.writeFileSync(path.join(dir, "index.html"), frame(post.title + " — Zhaoyuan Sun",
    post.summary, canonical, body, true, {
      "@context": "https://schema.org", "@type": "BlogPosting", headline: post.title,
      datePublished: post.published, dateModified: post.updated, inLanguage: "zh-CN",
      author: { "@type": "Person", name: "Zhaoyuan Sun", url: site },
      mainEntityOfPage: canonical
    }));
  reports.push({ slug: post.slug, math: compiled.math, displayMath: compiled.displayMath });
}
const entries = posts.map(post => [
  '<li><time datetime="' + post.date + '">' + post.date + "</time>",
  '<h2><a href="/blogs/' + post.slug + '/">' + escape(post.title) + "</a></h2>",
  "<p>" + escape(post.summary) + "</p></li>"
].join("")).join("\n");
fs.writeFileSync(path.join(root, "blogs/index.html"), frame("Blogs — Zhaoyuan Sun",
  "Technical notes on reasoning, learning, and robot agents.", site + "/blogs/",
  '<main><header class="index-header"><div class="eyebrow">Notes on models, learning, and robotics</div><h1>Blogs</h1>' +
  "<p>关于模型、学习与机器人执行的笔记。</p></header>" +
  '<ol class="post-list">' + entries + "</ol></main>", false));
console.log(JSON.stringify(reports));
