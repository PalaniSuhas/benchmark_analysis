import os

# Define paths
viz_dir = '/Users/kusugi23/Downloads/try/visualizations'
report_path = '/Users/kusugi23/Downloads/try/Web Search Analysis.pdf'
output_file = os.path.join(viz_dir, 'index.html')

# Get all html files
os.chdir(viz_dir)
files = [f for f in os.listdir('.') if f.endswith('.html') and f != 'index.html']
files.sort()

# Build the sidebar buttons
buttons_html = ""
for f in files:
    buttons_html += f'<button onclick="loadViz(\'{f}\', this)">{f}</button>\n'

# Add the PDF Report button at the end
# Note: Since the PDF is one folder up, we use '../' to reference it
if os.path.exists(report_path):
    buttons_html += '<hr style="border: 0; border-top: 1px solid #555; margin: 15px 0;">'
    buttons_html += f'<button onclick="loadViz(\'../Web Search Analysis.pdf\', this)" style="background: #2e7d32; color: white;">📄 View Report (PDF)</button>'

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Analysis Dashboard</title>
    <style>
        body {{ display: flex; margin: 0; font-family: -apple-system, sans-serif; height: 100vh; background: #222; }}
        #sidebar {{ width: 280px; background: #333; border-right: 1px solid #444; padding: 20px; overflow-y: auto; flex-shrink: 0; }}
        #content-area {{ flex-grow: 1; background: white; }}
        iframe {{ width: 100%; height: 100%; border: none; }}
        button {{ 
            display: block; width: 100%; text-align: left; padding: 12px; margin-bottom: 8px; 
            cursor: pointer; border: none; border-radius: 6px; background: #444; color: #ddd;
            transition: 0.2s; font-size: 14px;
        }}
        button:hover {{ background: #555; color: white; }}
        button.active {{ background: #007AFF; color: white; }}
        h2 {{ font-size: 1.1rem; margin-top: 0; color: #007AFF; text-transform: uppercase; letter-spacing: 1px; }}
    </style>
</head>
<body>

<div id="sidebar">
    <h2>Project Assets</h2>
    <div id="buttons">
        {buttons_html}
    </div>
</div>

<iframe id="viewer" src="{files[0] if files else ''}"></iframe>

<script>
    function loadViz(file, btn) {{
        document.getElementById('viewer').src = file;
        document.querySelectorAll('button').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
    }}
    // Highlight the first button on load
    document.querySelector('button').classList.add('active');
</script>

</body>
</html>
"""

with open(output_file, "w") as f:
    f.write(html_content)

print(f"Successfully created dashboard at {output_file}")