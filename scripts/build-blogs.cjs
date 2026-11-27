const fs = require("fs");
const path = require("path");
const { marked } = require("marked");
const katex = require("katex");

const root = path.resolve(__dirname, "..");
const site = "https://joesun-421.github.io";
const posts = JSON.parse(fs.readFileSync(path.join(root, "blogs/posts.json"), "utf8"));

const escape = value => String(value).replace(/[&<>"']/g, c =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

function readImageDimensions(filePath) {
  try {
    const fd = fs.openSync(filePath, "r");
    const buf = Buffer.alloc(24);
    const bytes = fs.readSync(fd, buf, 0, 24, 0);
    fs.closeSync(fd);
    // PNG signature + IHDR width/height
    if (bytes >= 24 && buf[0] === 0x89 && buf[1] === 0x50 && buf[2] === 0x4e && buf[3] === 0x47) {
      return { width: buf.readUInt32BE(16), height: buf.readUInt32BE(20) };
    }
  } catch (_) {
    // Fallback gracefully if non-PNG or missing
  }
  return null;
}

const FAVICON = "data:image/svg+xml," + encodeURIComponent(
  '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">' +
  '<rect width="64" height="64" fill="#0a0a0a"/>' +
  '<text x="50%" y="54%" dominant-baseline="middle" text-anchor="middle" ' +
  'fill="#ffffff" font-family="Georgia,serif" font-size="30" font-style="italic">ZS.</text></svg>'
);

const ARTICLE_SCRIPT = `(function(){
  var bar = document.querySelector(".reading-progress-bar");
  var backBtn = document.querySelector(".floating-top");
  var lastY = window.scrollY;
  var ticking = false;

  var headings = Array.prototype.slice.call(document.querySelectorAll(".prose h2[id], .prose h3[id]"));
  var tocLinks = Array.prototype.slice.call(document.querySelectorAll(".toc-rail a[href^='#']"));

  function updateSpy() {
    if (!headings.length || !tocLinks.length) return;
    var offset = 124;
    var current = headings[0].id;
    for (var i = 0; i < headings.length; i++) {
      if (headings[i].getBoundingClientRect().top <= offset) {
        current = headings[i].id;
      } else {
        break;
      }
    }
    if ((window.innerHeight + window.scrollY) >= document.documentElement.scrollHeight - 24) {
      current = headings[headings.length - 1].id;
    }
    tocLinks.forEach(function(a) {
      var active = a.getAttribute("href").slice(1) === current;
      a.classList.toggle("is-active", active);
      a.classList.remove("is-parent-active");
      if (active) a.setAttribute("aria-current", "location");
      else a.removeAttribute("aria-current");
    });
    var activeLink = document.querySelector(".toc-rail a.is-active");
    if (activeLink && activeLink.closest(".toc-sublist")) {
      var parentLi = activeLink.closest(".toc-sublist").parentElement;
      var parentLink = parentLi && parentLi.querySelector(":scope > .toc-sec-link");
      if (parentLink) parentLink.classList.add("is-parent-active");
    }
  }

  function onScroll() {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(function() {
      var y = window.scrollY;
      var doc = document.documentElement;
      var max = Math.max(1, doc.scrollHeight - window.innerHeight);
      var ratio = Math.min(1, Math.max(0, y / max));
      if (bar) bar.style.transform = "scaleX(" + ratio.toFixed(4) + ")";
      if (backBtn) {
        var scrollingUp = y < lastY - 4;
        var nearBottom = (window.innerHeight + y) >= doc.scrollHeight - 180;
        if (y <= 720 || nearBottom) backBtn.classList.remove("is-visible");
        else if (scrollingUp) backBtn.classList.add("is-visible");
        else if (y > lastY + 12) backBtn.classList.remove("is-visible");
      }
      lastY = y;
      updateSpy();
      ticking = false;
    });
  }

  var mathBlocks = Array.prototype.slice.call(document.querySelectorAll(".math-block"));
  function updateMathScrollCues() {
    mathBlocks.forEach(function(block) {
      var sc = block.querySelector(".math-scroll");
      if (!sc) return;
      var overflow = sc.scrollWidth > sc.clientWidth + 2;
      var atStart = sc.scrollLeft <= 2;
      var atEnd = sc.scrollLeft + sc.clientWidth >= sc.scrollWidth - 2;
      block.classList.toggle("is-scrollable", overflow);
      block.classList.toggle("is-scrolled-start", overflow && atStart);
      block.classList.toggle("is-scrolled-end", overflow && atEnd);
    });
  }
  mathBlocks.forEach(function(block) {
    var sc = block.querySelector(".math-scroll");
    if (sc) sc.addEventListener("scroll", updateMathScrollCues, { passive: true });
  });

  var tocMobile = document.querySelector(".toc-mobile");
  if (tocMobile) {
    tocMobile.querySelectorAll("a[href^='#']").forEach(function(link) {
      link.addEventListener("click", function() {
        tocMobile.removeAttribute("open");
      });
    });
  }

  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", function() {
    onScroll();
    updateMathScrollCues();
  }, { passive: true });
  window.addEventListener("beforeprint", function() {
    document.querySelectorAll('img[loading="lazy"]').forEach(function(img) {
      img.setAttribute("loading", "eager");
    });
  });
  onScroll();
  updateMathScrollCues();
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(updateMathScrollCues);
  }

  var dialog = document.getElementById("figure-lightbox");
  if (dialog && typeof dialog.showModal === "function") {
    var lbStage = dialog.querySelector(".lightbox-stage");
    var lbNum = dialog.querySelector(".lightbox-num");
    var lbCap = dialog.querySelector(".lightbox-caption");
    var lbZoom = dialog.querySelector(".lightbox-zoom");
    var lbClose = dialog.querySelector(".lightbox-close");

    function setZoom(zoomed) {
      dialog.classList.toggle("is-zoomed", zoomed);
      if (lbZoom) {
        lbZoom.setAttribute("aria-pressed", zoomed ? "true" : "false");
        lbZoom.textContent = zoomed ? "Fit" : "Zoom 2×";
      }
      if (lbStage) {
        lbStage.scrollLeft = 0;
        lbStage.scrollTop = 0;
      }
    }

    function openFigure(frame) {
      var img = frame.querySelector("img");
      var fig = frame.closest("figure");
      if (!img || !lbStage) return;
      var lbImg = lbStage.querySelector("img");
      if (!lbImg) {
        lbImg = document.createElement("img");
        lbImg.addEventListener("click", function() {
          setZoom(!dialog.classList.contains("is-zoomed"));
        });
        lbStage.appendChild(lbImg);
      }
      lbImg.src = img.currentSrc || img.src;
      lbImg.alt = img.alt || "";
      var numEl = fig && fig.querySelector(".fig-num");
      var capEl = fig && fig.querySelector(".fig-caption");
      lbNum.textContent = numEl ? numEl.textContent : "";
      lbCap.textContent = capEl ? capEl.textContent : (img.alt || "");
      setZoom(false);
      dialog.showModal();
    }

    document.querySelectorAll(".figure-frame").forEach(function(frame) {
      frame.addEventListener("click", function() { openFigure(frame); });
      frame.addEventListener("keydown", function(e) {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          openFigure(frame);
        }
      });
    });

    if (lbZoom) {
      lbZoom.addEventListener("click", function() {
        setZoom(!dialog.classList.contains("is-zoomed"));
      });
    }
    lbClose.addEventListener("click", function() { dialog.close(); });
    dialog.addEventListener("click", function(e) {
      if (e.target === dialog) dialog.close();
    });
    dialog.addEventListener("close", function() {
      setZoom(false);
      if (lbStage) lbStage.textContent = "";
    });
  }
})();`;

function frame(title, description, canonical, body, opts) {
  const { isArticle, isIndex, structured } = opts || {};
  const rootPrefix = isArticle ? "../../" : "../";
  const blogsIndex = isArticle ? "../index.html" : "index.html";
  const blogCss = isArticle ? "../blog.css" : "blog.css";
  return [
    '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">',
    '<meta name="viewport" content="width=device-width,initial-scale=1">',
    '<meta name="theme-color" content="#ffffff">',
    '<meta name="description" content="' + escape(description) + '">',
    '<meta property="og:type" content="' + (isArticle ? "article" : "website") + '">',
    '<meta property="og:title" content="' + escape(title) + '">',
    '<meta property="og:description" content="' + escape(description) + '">',
    '<meta property="og:url" content="' + canonical + '">',
    '<title>' + escape(title) + '</title>',
    '<link rel="icon" type="image/svg+xml" href="' + FAVICON + '">',
    '<link rel="canonical" href="' + canonical + '">',
    '<link rel="stylesheet" href="' + rootPrefix + 'assets/vendor/cormorant-garamond/font.css">',
    '<link rel="stylesheet" href="' + blogCss + '">',
    isArticle ? '<link rel="stylesheet" href="' + rootPrefix + 'assets/vendor/katex-0.19.1/katex.min.css">' : "",
    structured ? '<script type="application/ld+json">' +
      JSON.stringify(structured).replace(/</g, "\\u003c") + "</script>" : "",
    '</head><body class="' + (isArticle ? "page-post" : "page-index") + '">',
    isArticle ? '<div class="reading-progress" aria-hidden="true"><div class="reading-progress-bar"></div></div>' : "",
    '<div class="wrap">',
    '<header class="topbar" id="top"><a class="brand" href="' + rootPrefix + 'index.html" aria-label="Zhaoyuan Sun, home">ZS.</a>',
    '<nav class="nav" aria-label="Main navigation">' +
      '<a href="' + rootPrefix + 'index.html">Home</a>' +
      '<a href="' + blogsIndex + '"' + (isIndex ? ' aria-current="page"' : (isArticle ? ' class="is-section"' : "")) + '>Blogs</a>' +
      '<a href="' + rootPrefix + 'ideas.html">Ideas</a>' +
    '</nav></header>',
    body,
    '<footer class="site-footer">' +
      '<span class="footer-copy">© 2026 Zhaoyuan Sun</span>' +
      '<div class="footer-links">' +
        (isIndex
          ? '<a href="' + rootPrefix + 'index.html">Home</a><a href="' + rootPrefix + 'ideas.html">Ideas</a><a href="#top">Back to top ↑</a>'
          : '<a href="' + blogsIndex + '">All Blogs</a><a href="' + rootPrefix + 'index.html">Home</a><a href="#top">Back to top ↑</a>') +
      '</div>' +
    '</footer>',
    "</div>",
    isArticle ? [
      '<a class="floating-top" href="#top" aria-label="Back to top">↑</a>',
      '<dialog id="figure-lightbox" class="lightbox" aria-label="Figure viewer">',
      '<div class="lightbox-inner">',
      '<div class="lightbox-bar"><div class="lightbox-top"><span class="lightbox-num"></span>' +
        '<div class="lightbox-actions">' +
        '<button type="button" class="lightbox-zoom" aria-pressed="false" aria-label="Toggle figure zoom">Zoom 2×</button>' +
        '<button type="button" class="lightbox-close" aria-label="Close figure viewer">Close ×</button>' +
        '</div></div><div class="lightbox-caption"></div></div>',
      '<div class="lightbox-stage"></div>',
      '</div></dialog>',
      '<script>' + ARTICLE_SCRIPT + '</script>'
    ].join("\n") : "",
    "</body></html>"
  ].filter(Boolean).join("\n");
}

function estimateReadingMinutes(markdown) {
  const clean = markdown
    .replace(/^---\n[\s\S]*?\n---\n/, "")
    .replace(/\$\$[\s\S]*?\$\$/g, " ")
    .replace(/\$[^\$\n]+?\$/g, " ")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, " ")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1");
  const cjk = (clean.match(/[\u4e00-\u9fff]/g) || []).length;
  const latin = (clean.replace(/[\u4e00-\u9fff]/g, " ").match(/[A-Za-z0-9][A-Za-z0-9._-]*/g) || []).length;
  return Math.max(1, Math.round(cjk / 320 + latin / 200));
}

function compileMarkdown(source, postDir) {
  const readingMinutes = estimateReadingMinutes(source);
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

  text = text.replace(/^[ \t]*\$\$[ \t]*\r?\n([\s\S]*?)\r?\n[ \t]*\$\$[ \t]*$/gm,
    (_, tex) => "\n\n" + protect(tex.trim(), true) + "\n\n");
  text = text.replace(/^[ \t]*\$\$[ \t]*((?:[^$\n]|\\\$)+?)[ \t]*\$\$[ \t]*$/gm,
    (_, tex) => "\n\n" + protect(tex.trim(), true) + "\n\n");
  text = text.replace(/(?<!\\)\$([^\$\n]+?)(?<!\\)\$/g,
    (_, tex) => protect(tex, false));

  let html = marked.parse(text, { gfm: true });

  let displayIdx = 0;
  formulae.forEach((f, i) => {
    const key = "BLOGMATHPLACEHOLDER" + i + "END";
    if (f.display) {
      displayIdx += 1;
      const blockHtml =
        '<div class="math-block" data-eq="' + displayIdx + '">' +
        '<div class="math-viewport"><div class="math-scroll">' + f.html + '</div></div>' +
        '<span class="math-num" aria-hidden="true">(' + displayIdx + ')</span></div>';
      if (html.includes("<p>" + key + "</p>")) {
        html = html.replace("<p>" + key + "</p>", blockHtml);
      } else {
        html = html.replace(key, blockHtml);
      }
    } else {
      html = html.replaceAll(key, '<span class="math-inline">' + f.html + "</span>");
    }
  });
  if (html.includes("BLOGMATHPLACEHOLDER")) throw new Error("Unrendered formula");

  // Wrap rendered tables in scroll container so tables never cause horizontal overflow on mobile
  html = html.replace(/<table(\b[^>]*)>([\s\S]*?)<\/table>/g, '<div class="table-scroll"><table$1>$2</table></div>');

  // Transform standalone images into numbered exhibition-framed figures with intrinsic dimensions (zero CLS)
  let figureCount = 0;
  html = html.replace(/<p><img src="([^"]+)" alt="([^"]*)"><\/p>/g, (_, src, alt) => {
    figureCount += 1;
    const num = String(figureCount).padStart(2, "0");
    const dims = !/^https?:\/\//.test(src) ? readImageDimensions(path.resolve(postDir, src)) : null;
    const dimAttrs = dims ? ' width="' + dims.width + '" height="' + dims.height + '"' : "";
    return [
      '<figure class="figure" id="fig-' + figureCount + '">',
      '<div class="figure-frame" tabindex="0" role="button" aria-label="Zoom Figure ' + num + (alt ? ": " + alt : "") + '">',
      '<img src="' + src + '" alt="' + alt + '"' + dimAttrs + ' loading="lazy" decoding="async">',
      "</div>",
      alt ? '<figcaption><span class="fig-num">Fig. ' + num + '</span><span class="fig-caption">' + alt + "</span></figcaption>" : "",
      "</figure>"
    ].filter(Boolean).join("");
  });

  // Style standalone bold sub-topic lead-in paragraphs (e.g., **Inference-Time Exit**)
  html = html.replace(/<p><strong>([^<]+)<\/strong><\/p>/g, '<p class="subtopic-lead"><strong>$1</strong></p>');

  // Process H2 / H3 headings: preserve exact slug IDs, add section numbers, split H3 paper titles, build nested TOC
  const outline = [];
  const seen = new Map();
  let h2Count = 0;
  let paperCount = 0;

  html = html.replace(/<h([23])>([\s\S]*?)<\/h\1>/g, (_, level, label) => {
    const plain = label.replace(/<[^>]+>/g, "");
    const base = plain.toLowerCase().replace(/[^\p{L}\p{N} -]/gu, "").trim().replace(/\s+/g, "-");
    const n = seen.get(base) || 0;
    seen.set(base, n + 1);
    const id = base + (n ? "-" + n : "");

    if (level === "2") {
      h2Count += 1;
      const num = String(h2Count).padStart(2, "0");
      const h2DashIdx = plain.indexOf(" — ");
      const shortLabel = h2DashIdx > 0 ? plain.slice(0, h2DashIdx).trim() : plain;
      outline.push({ id, label: plain, shortLabel, num, children: [] });
      return (
        '<h2 id="' + id + '">' +
        '<span class="sec-num" aria-hidden="true">' + num + "</span>" +
        '<span class="heading-text">' + label +
        '<a class="heading-anchor" href="#' + id + '" aria-label="Link to section: ' + escape(plain) + '">#</a>' +
        "</span></h2>"
      );
    }

    paperCount += 1;
    const dashIdx = plain.indexOf(" — ");
    const shortLabel = dashIdx > 0 ? plain.slice(0, dashIdx).trim() : plain;
    if (outline.length) {
      outline[outline.length - 1].children.push({ id, label: plain, shortLabel });
    }
    let formatted = label;
    const rawDashIdx = label.indexOf(" — ");
    if (rawDashIdx > 0) {
      const mainPart = label.slice(0, rawDashIdx);
      const subPart = label.slice(rawDashIdx + 3);
      formatted = '<span class="h3-main">' + mainPart + '</span><span class="h3-sep" aria-hidden="true"> — </span><span class="h3-sub">' + subPart + "</span>";
    }
    return (
      '<h3 id="' + id + '">' +
      '<span class="heading-text">' + formatted +
      '<a class="heading-anchor" href="#' + id + '" aria-label="Link to subsection: ' + escape(plain) + '">#</a>' +
      "</span></h3>"
    );
  });

  // Style paper metadata lines immediately following H3 paper headings
  html = html.replace(/(<h3\b[^>]*>[\s\S]*?<\/h3>\s*)<p><em>((?:(?!<\/em><\/p>)[\s\S])*)<\/em><\/p>/g, '$1<p class="paper-meta">$2</p>');

  // Wrap References block cleanly for dedicated academic citation typography
  const refMarker = '<h2 id="references">';
  const refIndex = html.indexOf(refMarker);
  if (refIndex !== -1) {
    html =
      html.slice(0, refIndex) +
      '<section class="references-section" aria-labelledby="references">' +
      html.slice(refIndex) +
      "</section>";
  }

  return {
    html,
    outline,
    math: formulae.length,
    displayMath: displayIdx,
    figureCount,
    paperCount,
    readingMinutes
  };
}

function renderTocList(outline) {
  return '<ol class="toc-list">' + outline.map(sec => {
    const sub = sec.children.length
      ? '<ul class="toc-sublist">' + sec.children.map(c =>
          '<li><a href="#' + c.id + '">' + escape(c.shortLabel) + "</a></li>"
        ).join("") + "</ul>"
      : "";
    return (
      '<li><a class="toc-sec-link" href="#' + sec.id + '" title="' + escape(sec.label) + '">' +
      '<span class="toc-num">' + sec.num + '</span>' +
      '<span class="toc-label">' + escape(sec.shortLabel || sec.label) + "</span></a>" + sub + "</li>"
    );
  }).join("") + "</ol>";
}

const reports = [];
const postStats = new Map();

for (const post of posts) {
  const dir = path.join(root, "blogs", post.slug);
  const source = fs.readFileSync(path.join(dir, "source.md"), "utf8");
  const compiled = compileMarkdown(source, dir);
  postStats.set(post.slug, compiled);

  const canonical = site + "/blogs/" + post.slug + "/";
  const tocMarkup = renderTocList(compiled.outline);

  const statsParts = [compiled.readingMinutes + " min read"];
  if (compiled.figureCount) statsParts.push(compiled.figureCount + " figures");
  if (compiled.displayMath) statsParts.push(compiled.displayMath + " equations");

  const metaGroups = [
    '<span class="meta-group"><time datetime="' + post.date + '">' + post.date + '</time><span class="meta-dot" aria-hidden="true">·</span><span>Zhaoyuan Sun</span></span>',
    '<span class="meta-group">' + statsParts.map(s => "<span>" + s + "</span>").join('<span class="meta-dot" aria-hidden="true">·</span>') + "</span>",
    '<span class="meta-group"><a href="https://github.com/JoeSun-421/joesun-421.github.io/blob/main/blogs/' + post.slug + '/source.md">Markdown source ↗</a></span>'
  ].join("");

  const body = [
    '<main class="post-shell">',
    '<header class="post-header">',
    '<div class="eyebrow"><span class="eyebrow-mark" aria-hidden="true"></span>Research notes · Adaptive reasoning</div>',
    "<h1>" + escape(post.title) + "</h1>",
    post.summary ? '<p class="post-lead">' + escape(post.summary) + "</p>" : "",
    '<div class="post-meta">' + metaGroups + "</div>",
    "</header>",
    '<div class="article-layout">',
    '<aside class="toc-rail" aria-label="Table of contents">',
    '<div class="toc-rail-inner">',
    '<div class="toc-title">Contents</div>',
    '<nav class="toc-nav">' + tocMarkup + "</nav>",
    '<a class="toc-top" href="#top">↑ Back to top</a>',
    "</div></aside>",
    '<div class="article-main">',
    '<details class="toc-mobile">',
    '<summary><span>Contents</span><span class="toc-summary-meta">' + compiled.outline.length + " sections</span></summary>",
    '<nav class="toc-nav">' + tocMarkup + "</nav>",
    "</details>",
    '<article class="prose">' + compiled.html + "</article>",
    "</div></div></main>"
  ].filter(Boolean).join("\n");

  fs.writeFileSync(path.join(dir, "index.html"), frame(
    post.title + " — Zhaoyuan Sun",
    post.summary,
    canonical,
    body,
    {
      isArticle: true,
      isIndex: false,
      structured: {
        "@context": "https://schema.org", "@type": "BlogPosting", headline: post.title,
        datePublished: post.published, dateModified: post.updated, inLanguage: "zh-CN",
        author: { "@type": "Person", name: "Zhaoyuan Sun", url: site },
        mainEntityOfPage: canonical
      }
    }
  ));

  reports.push({
    slug: post.slug,
    math: compiled.math,
    displayMath: compiled.displayMath,
    figures: compiled.figureCount,
    papers: compiled.paperCount,
    readingMinutes: compiled.readingMinutes
  });
}

const entries = posts.map(post => {
  const st = postStats.get(post.slug);
  const idxRows = [];
  if (st) idxRows.push('<span class="post-item-stats">' + st.readingMinutes + " min read</span>");
  const countParts = [];
  if (st && st.figureCount) countParts.push(st.figureCount + " figures");
  if (st && st.displayMath) countParts.push(st.displayMath + " equations");
  if (countParts.length) idxRows.push('<span class="post-item-stats">' + countParts.join(" · ") + "</span>");
  const subMeta = idxRows.join("");
  return [
    '<li class="post-item">',
    '<a class="post-card" href="' + post.slug + '/index.html">',
    '<div class="post-item-aside"><time datetime="' + post.date + '">' + post.date + "</time>" + subMeta + "</div>",
    '<div class="post-item-body">',
    "<h2>" + escape(post.title) + "</h2>",
    "<p>" + escape(post.summary) + "</p>",
    '<span class="post-read-more">Read essay <span aria-hidden="true">→</span></span>',
    "</div></a></li>"
  ].join("\n");
}).join("\n");

fs.writeFileSync(path.join(root, "blogs/index.html"), frame(
  "Blogs — Zhaoyuan Sun",
  "Technical notes on reasoning, learning, and robot agents.",
  site + "/blogs/",
  '<main class="index-shell"><header class="index-header">' +
  '<div class="eyebrow"><span class="eyebrow-mark" aria-hidden="true"></span>Notes on models, learning, and robotics</div>' +
  "<h1>Blogs</h1>" +
  "<p>关于模型、学习与机器人执行的系统性研究笔记。</p></header>" +
  '<ol class="post-list">' + entries + "</ol></main>",
  { isArticle: false, isIndex: true }
));

console.log(JSON.stringify(reports));
