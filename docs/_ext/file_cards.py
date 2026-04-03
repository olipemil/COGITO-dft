"""
file_cards — reads docs/file_struc/<Module>/{input,output}.file.

Provides two directives:

  :::{file-cards} COGITO input
  :::
      Renders a sphinx-inline-tabs tab-set, one tab per @file{} entry.
      Tab group ID is deterministic: fc-<Module>-<filetype>
      Tab input IDs:                 fc-<Module>-<filetype>-<index>

  :::{file-flow} COGITO
  :::
      Renders a left→middle→right flow diagram.
      Each box carries data-tab-target pointing at the matching tab input,
      so clicking a box activates the corresponding tab below.
"""
import os, re
from docutils import nodes
from docutils.parsers.rst import Directive


# ── parser ──────────────────────────────────────────────────────────────────

def parse_file_spec(text):
    entries = []
    for m in re.finditer(r'@file\{([^,\n]+),(.*?)\}', text, re.DOTALL):
        entry = {'id': m.group(1).strip()}
        for kv in re.finditer(r'(\w+)\s*=\s*(".*?"|\'.*?\'|\[.*?\]|[^\n,]+)', m.group(2)):
            key, val = kv.group(1), kv.group(2).strip().strip(',')
            if val.startswith('"') or val.startswith("'"):
                val = val[1:-1]
            elif val.startswith('['):
                val = re.findall(r"['\"]([^'\"]+)['\"]", val)
            entry[key] = val
        entries.append(entry)
    return entries


def _load(srcdir, module, filetype, env):
    path = os.path.join(srcdir, 'file_struc', module, f'{filetype}.file')
    if not os.path.exists(path):
        return []
    env.note_dependency(path)
    return parse_file_spec(open(path).read())


# ── shared helpers ───────────────────────────────────────────────────────────

def _card_content_html(entry, module, filetype, srcdir='', depth=0):
    tags = entry.get('tags', [])
    if isinstance(tags, str):
        tags = [tags]
    tag_html = ''.join(f'<span class="file-tag {t}">{t}</span>' for t in tags)

    rows = ''

    run_with = entry.get('run_with', [])
    if isinstance(run_with, str):
        run_with = [run_with] if run_with.strip() else []
    if run_with:
        label = 'Used by:' if filetype == 'input' else 'Made by:'
        funcs_html = ''
        for func in run_with:
            func = func.strip()
            func_name = re.sub(r'\(.*\)$', '', func)
            api_href  = f'api/{module}.html#COGITO_dft.{module}.{func_name}'
            funcs_html += (
                f'{func}'
                f' <a class="file-api-link" href="{api_href}">[api]</a><br>'
            )
        rows += (
            f'<span class="file-card-info-label">{label}</span>'
            f'<span class="file-card-info-value">{funcs_html}</span>\n'
        )

    for row_label, key in [
        ('Requires:',    'needed_tags'),
        ('Make in CLI:', 'run_CLI'),
        ('Skip in CLI:', 'skip_CLI'),
    ]:
        val = entry.get(key, '').strip()
        if val:
            rows += (
                f'<span class="file-card-info-label">{row_label}</span>'
                f'<span class="file-card-info-value">{val}</span>\n'
            )

    desc = entry.get('more_info', '').strip()
    desc_html = ''
    if desc:
        desc_html = (
            f'<div class="file-card-description">'
            f'<span class="file-card-info-label">Description</span>'
            f'<div class="file-card-desc-text">{desc}</div>'
            f'</div>'
        )

    preview = _preview_html(entry.get('file_img', '').strip(), srcdir, depth)

    left = (
        f'<div class="file-card-tags">{tag_html}</div>'
        f'<div class="file-card-info">{rows}</div>'
        + desc_html
    )
    if preview:
        return (
            f'<div class="file-card-body">'
            f'<div class="file-card-left">{left}</div>'
            f'<div class="file-card-right">{preview}</div>'
            f'</div>'
        )
    return left


def _box_style(tags):
    if 'image' in tags:
        color = '#2471a3'
    elif 'quality' in tags:
        color = '#c0392b'
    elif 'debug' in tags:
        color = '#b7950b'
    else:
        color = 'var(--color-foreground-primary)'
    bw = '2.5px' if 'required' in tags else '1.5px'
    fw = 'bold'  if 'required' in tags else 'normal'
    return color, bw, fw


# ── preview helpers ───────────────────────────────────────────────────────────

def _escape_html(text):
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _json_tree_lines(obj, prefix='', depth=0, max_depth=3, max_items=6):
    lines = []
    # Only truncate at the top layer; show all items at deeper levels
    limit = max_items if depth == 0 else None
    if isinstance(obj, dict):
        items = list(obj.items())
        n = len(items)
        shown = items[:limit] if limit else items
        for idx, (k, v) in enumerate(shown):
            is_last = (idx == len(shown) - 1) and (limit is None or n <= limit)
            connector = '└── ' if is_last else '├── '
            child_prefix = prefix + ('    ' if is_last else '│   ')
            lines.append(prefix + connector + str(k))
            if depth < max_depth:
                if isinstance(v, dict):
                    lines.extend(_json_tree_lines(v, child_prefix, depth + 1, max_depth, max_items))
                elif isinstance(v, list) and v and isinstance(v[0], dict):
                    lines.extend(_json_tree_lines(v, child_prefix, depth + 1, max_depth, max_items))
        if limit and n > limit:
            lines.append(prefix + '└── ...')
    elif isinstance(obj, list):
        n = len(obj)
        shown = obj[:limit] if limit else obj
        for idx, v in enumerate(shown):
            is_last = (idx == len(shown) - 1) and (limit is None or n <= limit)
            connector = '└── ' if is_last else '├── '
            child_prefix = prefix + ('    ' if is_last else '│   ')
            lines.append(prefix + connector + f'[{idx}]')
            if isinstance(v, dict) and depth < max_depth:
                lines.extend(_json_tree_lines(v, child_prefix, depth + 1, max_depth, max_items))
        if limit and n > limit:
            lines.append(prefix + '└── ...')
    return lines


def _preview_html(img_src, srcdir, depth=0):
    """Return preview element HTML for a file_img entry (always visible, no toggle).

    PNG/JPG are embedded as base64 data URIs (no path dependency).
    HTML iframes use a depth-corrected relative URL (html_extra_path copies
    PbO/ contents to the build root, so only the basename is needed).
    TXT/JSON are read at build time and embedded inline.
    """
    if not img_src:
        return ''
    ext = os.path.splitext(img_src)[1].lower()
    full = os.path.join(srcdir, img_src)

    if ext in ('.png', '.jpg', '.jpeg', '.gif'):
        try:
            import base64
            mime = {'png': 'image/png', 'jpg': 'image/jpeg',
                    'jpeg': 'image/jpeg', 'gif': 'image/gif'}.get(ext.lstrip('.'), 'image/png')
            with open(full, 'rb') as fh:
                data = base64.b64encode(fh.read()).decode()
        except OSError:
            return ''
        return f'<img class="file-card-preview" src="data:{mime};base64,{data}" alt="example output">'

    if ext == '.html':
        # html_extra_path copies PbO/ contents to the build root, so only the
        # basename is needed; depth gives the number of ../ required.
        basename = os.path.basename(img_src)
        rel = '../' * depth + basename
        return (
            f'<iframe class="file-card-preview file-card-iframe"'
            f' data-src="{rel}" src="about:blank"></iframe>'
        )

    if ext == '.txt':
        try:
            with open(full, encoding='utf-8', errors='replace') as fh:
                raw_lines = fh.readlines()
            text = ''.join(raw_lines[:30]) + ('' if len(raw_lines) <= 30 else '\n...')
        except OSError:
            return ''
        return f'<pre class="file-card-preview file-card-pre">{_escape_html(text)}</pre>'

    if ext == '.json':
        try:
            import json
            with open(full, encoding='utf-8') as fh:
                obj = json.load(fh)
            tree = '\n'.join(['root'] + _json_tree_lines(obj))
        except (OSError, ValueError):
            return ''
        return f'<pre class="file-card-preview file-card-pre">{_escape_html(tree)}</pre>'

    return ''


# ── file-cards directive ─────────────────────────────────────────────────────

class FileCardsDirective(Directive):
    required_arguments = 1
    optional_arguments = 1   # filetype: 'input' or 'output' (default: output)
    has_content = False

    def run(self):
        module   = self.arguments[0]
        filetype = self.arguments[1] if len(self.arguments) > 1 else 'output'
        env      = self.state.document.settings.env
        entries  = _load(env.srcdir, module, filetype, env)

        if not entries:
            return [nodes.warning('', nodes.paragraph(
                text=f'No {filetype}.file found for {module}'))]

        group = f'fc-{module}-{filetype}'

        labels_html   = ''
        contents_html = ''
        for i, entry in enumerate(entries):
            tab_id  = f'{group}-{i}'
            checked = 'checked' if i == 0 else ''
            name    = entry.get('file_name', entry['id'])
            depth   = env.docname.count('/')
            content = _card_content_html(entry, module, filetype, env.srcdir, depth)
            labels_html += (
                f'<input class="tab-input" id="{tab_id}" name="{group}" '
                f'type="radio" {checked}>'
                f'<label class="tab-label" for="{tab_id}">{name}</label>'
            )
            shown = 'block' if i == 0 else 'none'
            contents_html += (
                f'<div class="tab-content" data-tab="{tab_id}" '
                f'style="display:{shown}">{content}</div>\n'
            )

        html = (
            f'<div class="tab-set">'
            f'<div class="tab-label-bar">{labels_html}</div>'
            f'{contents_html}'
            f'</div>'
        )
        return [nodes.raw('', html, format='html')]


# ── file-flow directive ──────────────────────────────────────────────────────

_CLICK_SYNC_JS = """\
<script>
if (!window._fileFlowClickDefined) {
  window._fileFlowClickDefined = true;

  /* Lazy-load iframes in a newly visible tab panel */
  var _lazyFrames = function(panel) {
    if (!panel) return;
    panel.querySelectorAll('iframe.file-card-iframe[data-src]').forEach(function(fr) {
      if (fr.getAttribute('src') === 'about:blank') {
        fr.setAttribute('src', fr.dataset.src);
      }
    });
  };

  /* Shared: show the panel for tabId, scroll its label into view, lazy-load iframes */
  var _fcSwitch = function(tabId, tabSet) {
    if (!tabSet) return;
    var visible = null;
    tabSet.querySelectorAll('.tab-content').forEach(function(c) {
      var show = c.dataset.tab === tabId;
      c.style.display = show ? 'block' : 'none';
      if (show) visible = c;
    });
    _lazyFrames(visible);
    var bar = tabSet.querySelector('.tab-label-bar');
    var lbl = bar ? bar.querySelector('label[for="' + tabId + '"]') : null;
    if (bar && lbl) {
      var ll = lbl.offsetLeft, lr = ll + lbl.offsetWidth;
      if (lr > bar.scrollLeft + bar.clientWidth) bar.scrollLeft = lr - bar.clientWidth;
      else if (ll < bar.scrollLeft) bar.scrollLeft = ll;
    }
  };

  document.addEventListener('click', function(e) {
    /* Flow box click → activate corresponding tab */
    var box = e.target.closest('.file-flow-box[data-tab-target]');
    if (box) {
      var inp = document.getElementById(box.dataset.tabTarget);
      if (inp) { inp.checked = true; _fcSwitch(inp.id, inp.closest('.tab-set')); }
      return;
    }
    /* Tab label click → switch content panel */
    var lbl = e.target.closest('.tab-label-bar > label');
    if (lbl) {
      _fcSwitch(lbl.getAttribute('for'), lbl.closest('.tab-set'));
    }
  });

  /* On load: lazy-load iframes in the initially visible tab of each tab-set */
  document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.tab-set').forEach(function(ts) {
      var first = ts.querySelector('.tab-content[style*="display:block"]');
      _lazyFrames(first);
    });
  });
}
</script>
"""


class FileFlowDirective(Directive):
    required_arguments = 1
    optional_arguments = 0
    has_content = False

    def run(self):
        module  = self.arguments[0]
        env     = self.state.document.settings.env
        inputs  = _load(env.srcdir, module, 'input',  env)
        outputs = _load(env.srcdir, module, 'output', env)

        def boxes_html(entries, filetype):
            html = ''
            for i, entry in enumerate(entries):
                tags = entry.get('tags', [])
                if isinstance(tags, str):
                    tags = [tags]
                color, bw, fw = _box_style(tags)
                name      = entry.get('file_name', entry['id'])
                tab_id    = f'fc-{module}-{filetype}-{i}'
                style     = (
                    f'border:{bw} solid {color};'
                    f'color:{color};'
                    f'font-weight:{fw};'
                )
                html += (
                    f'<div class="file-flow-box" '
                    f'style="{style}" '
                    f'data-tab-target="{tab_id}">{name}</div>\n'
                )
            return html

        inputs_html  = boxes_html(inputs,  'input')
        outputs_html = boxes_html(outputs, 'output')

        html = (
            f'<div class="file-flow">'
            f'<div class="file-flow-col file-flow-inputs">{inputs_html}</div>'
            f'<div class="file-flow-arrow">\u2192</div>'
            f'<div class="file-flow-col file-flow-center">'
            f'<div class="file-flow-module-name">{module}</div>'
            f'</div>'
            f'<div class="file-flow-arrow">\u2192</div>'
            f'<div class="file-flow-col file-flow-outputs">{outputs_html}</div>'
            f'</div>'
            + _CLICK_SYNC_JS
        )
        return [nodes.raw('', html, format='html')]


# ── extension entry point ────────────────────────────────────────────────────

def setup(app):
    app.add_directive('file-cards', FileCardsDirective)
    app.add_directive('file-flow',  FileFlowDirective)
    return {'version': '0.1', 'parallel_read_safe': True}
