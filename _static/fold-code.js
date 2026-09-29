// Fold long code blocks marked with `:class: fold-code`: show the first lines,
// fade the rest, and expand in place with a button.
document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll("div.fold-code").forEach(function (block) {
    var pre = block.querySelector("pre");
    if (!pre || pre.scrollHeight < 420) return;
    block.classList.add("folded");
    var btn = document.createElement("button");
    btn.className = "fold-code-btn";
    btn.textContent = "Show full kernel";
    btn.addEventListener("click", function () {
      var folded = block.classList.toggle("folded");
      btn.textContent = folded ? "Show full kernel" : "Show less";
      if (folded) block.scrollIntoView({ block: "nearest" });
    });
    block.after(btn);
  });
});
