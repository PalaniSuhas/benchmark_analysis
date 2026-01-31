import os

# Use relative paths so it works anywhere (Local or GitHub)
viz_dir = '.' 
report_path = '../Web Search Analysis.pdf'
output_file = 'index.html'

# Get all html files in the current directory
files = [f for f in os.listdir(viz_dir) if f.endswith('.html') and f != 'index.html']
files.sort()

# Build the sidebar buttons
buttons_html = ""
for f in files:
    # Clean up the filename for the display label (e.g., "1_performance" -> "Performance")
    display_name = f.replace('_', ' ').replace('.html', '').title()
    buttons_html += f'<button onclick="loadViz(\'{f}\', this)">{display_name}</button>\n'

# Add the PDF Report button if it exists relative to this folder
if os.path.exists(report_path):
    buttons_html += '<hr style="border: 0; border-top: 1px solid #555; margin: 15px 0;">'
    buttons_html += f'<button onclick="loadViz(\'{report_path}\', this)" style="background: #2e7d32; color: white;">📄 View Report (PDF)</button>'

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Benchmark Analysis Dashboard</title>
    <style>
        body {{ display: flex; margin: 0; font-family: -apple-system, sans-serif; height: 100vh; background: #222; overflow: hidden; }}
        #sidebar {{ width: 280px; background: #333; border-right: 1px solid #444; padding: 20px; overflow-y: auto; flex-shrink: 0; }}
        #viewer-container {{ flex-grow: 1; background: white; position: relative; }}
        iframe {{ width: 100%; height: 100%; border: none; }}
        button {{ 
            display: block; width: 100%; text-align: left; padding: 12px; margin-bottom: 8px; 
            cursor: pointer; border: none; border-radius: 6px; background: #444; color: #ddd;
            transition: 0.2s; font-size: 13px; line-height: 1.4;
        }}
        button:hover {{ background: #555; color: white; }}
        button.active {{ background: #007AFF; color: white; font-weight: bold; }}
        h2 {{ font-size: 0.9rem; margin-top: 0; color: #888; text-transform: uppercase; letter-spacing: 2px; }}
    </style>
</head>
<body>

<div id="sidebar">
    <h2>Benchmarks</h2>
    <div id="buttons">
        {buttons_html}
    </div>
</div>

<div id="viewer-container">
    <iframe id="viewer" src="{files[0] if files else ''}"></iframe>
</div>

<script>
    function loadViz(file, btn) {{
        document.getElementById('viewer').src = file;
        let btns = document.querySelectorAll('button');
        btns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
    }}
    // Initialize first button
    window.onload = function() {{
        const firstBtn = document.querySelector('button');
        if (firstBtn) firstBtn.classList.add('active');
    }};
</script>

</body>
</html>
"""

with open(output_file, "w") as f:
    f.write(html_content)

print(f"Portable dashboard created: {len(files)} visuals indexed.")