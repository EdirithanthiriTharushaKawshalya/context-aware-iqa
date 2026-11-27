"""Compact controls with transparent edges and restrained photo-workspace tabs."""
from PIL import Image, ImageDraw, ImageTk


def install(root, style, mode='light'):
    host = root._root()
    from .appearance import DARK
    prefix = 'SoftDark.' if mode == 'dark' else 'Soft.'
    if not hasattr(host, '_soft_images'):
        host._soft_images = []
    panel = '#26272e' if mode == 'dark' else '#ffffff'
    workspace = '#1c1d23' if mode == 'dark' else '#f4f3f8'
    text = '#eeeaf8' if mode == 'dark' else '#242332'
    if prefix + 'card' not in style.element_names():
        def photo(image):
            result = ImageTk.PhotoImage(image, master=root)
            host._soft_images.append(result)
            return result
        def tile(fill, outline=None, radius=5, size=(40, 28)):
            if mode == 'dark':
                fill, outline = DARK.get(fill, fill), DARK.get(outline, outline)
            scale = 3
            image = Image.new('RGBA', (size[0]*scale, size[1]*scale))
            ImageDraw.Draw(image).rounded_rectangle(
                (1, 1, image.width-2, image.height-2), radius=radius*scale,
                fill=fill, outline=outline or fill, width=scale)
            return photo(image.resize(size, Image.Resampling.LANCZOS))
        def element(name, normal, *states, border=6):
            style.element_create(name, 'image', normal, *states, border=border, sticky='nswe')
        element(prefix+'card', tile('#ffffff', radius=8, size=(40, 32)), border=9)
        for name, normal, hover, pressed, disabled in (
            ('button', '#ffffff', '#f0eafa', '#e3d8f8', '#f4f1f8'),
            ('primary', '#7655d3', '#6847c4', '#5636ae', '#e7e0f3'),
            ('danger', '#fff2f5', '#fbe4eb', '#f5d2dd', '#f7f1f4')):
            element(prefix+name, tile(normal, '#e1dbe9'),
                    ('disabled', tile(disabled, '#e7e1ee')),
                    ('pressed', tile(pressed)), ('active', tile(hover)),
                    ('focus', tile(normal, '#b09ae9')))
        element(prefix+'field', tile('#faf8fd', '#e0daeb'),
                ('disabled', tile('#f4f1f7', '#e6e1ed')),
                ('focus', tile('#ffffff', '#a78cde')))
        arrow = Image.new('RGBA', (18, 18))
        ImageDraw.Draw(arrow).line([(5, 7), (9, 11), (13, 7)], fill='#a69db8' if mode == 'dark' else '#827397', width=2)
        style.element_create(prefix+'arrow', 'image', photo(arrow), sticky='')
        element(prefix+'trough', tile('#eee8f6', radius=3, size=(24, 6)), border=3)
        element(prefix+'pbar', tile('#aa91e6', radius=3, size=(24, 6)), border=3)
        for orient, size in (('Vertical', (10, 32)), ('Horizontal', (32, 10))):
            element(prefix+orient+'.Scrollbar.thumb', tile('#6f6b7b' if mode == 'dark' else '#c3bece', radius=4, size=size),
                    ('active', tile('#938b9f' if mode == 'dark' else '#a69ab9', radius=4, size=size)),
                    ('pressed', tile('#aa91e6', radius=4, size=size)),
                    ('disabled', tile(panel, radius=4, size=size)), border=4)
        def tab(fill, accent=None):
            image = Image.new('RGBA', (40, 28), fill)
            if accent:
                ImageDraw.Draw(image).rectangle((0, 26, 39, 27), fill=accent)
            return photo(image)
        element(prefix+'Notebook.tab', tab(panel),
                ('selected', tab('#32303d' if mode == 'dark' else '#f0eafa', '#b29af5' if mode == 'dark' else '#7655d3')),
                ('active', tab('#2d2d35' if mode == 'dark' else '#f7f5fb')), border=(6, 0, 6, 3))
    style.layout('Rounded.TFrame', [(prefix+'card', {'sticky': 'nswe'})])
    style.configure('Rounded.TFrame', background=workspace)
    for name, element, surface in (
        ('TButton', 'button', workspace), ('Primary.TButton', 'primary', workspace),
        ('Danger.TButton', 'danger', panel), ('Card.TButton', 'button', panel),
        ('Card.Primary.TButton', 'primary', panel)):
        style.layout(name, [(prefix+element, {'sticky': 'nswe', 'children': [
            ('Button.padding', {'sticky': 'nswe', 'children': [('Button.label', {'sticky': 'nswe'})]})]})])
        style.configure(name, background=surface, padding=(10, 5), borderwidth=0, font=('Segoe UI', 9))
        # The image owns hover fills. The widget background stays the parent color.
        style.map(name, background=[('disabled', surface), ('!disabled', surface)])
    style.configure('Primary.TButton', font=('Segoe UI', 9, 'bold'))
    style.configure('Photo.Card.TButton', padding=(8, 4))
    style.configure('Photo.Danger.TButton', padding=(8, 4))
    style.layout('TEntry', [(prefix+'field', {'sticky': 'nswe', 'children': [
        ('Entry.padding', {'sticky': 'nswe', 'children': [('Entry.textarea', {'sticky': 'nswe'})]})]})])
    style.layout('TCombobox', [(prefix+'field', {'sticky': 'nswe', 'children': [
        (prefix+'arrow', {'side': 'right', 'sticky': 'ns'}),
        ('Combobox.padding', {'sticky': 'nswe', 'children': [('Combobox.textarea', {'sticky': 'nswe'})]})]})])
    style.configure('TEntry', padding=(8, 5), background=panel)
    style.configure('TCombobox', padding=(8, 5), background=panel)
    style.layout('Horizontal.TProgressbar', [(prefix+'trough', {'sticky': 'nswe', 'children': [
        (prefix+'pbar', {'side': 'left', 'sticky': 'ns'})]})])
    style.configure('Horizontal.TProgressbar', thickness=6, borderwidth=0, background=panel)
    for orient in ('Vertical', 'Horizontal'):
        style.layout(orient+'.TScrollbar', [(orient+'.Scrollbar.trough', {'sticky': 'nswe', 'children': [
            (prefix+orient+'.Scrollbar.thumb', {'sticky': 'nswe', 'expand': True})]})])
        style.configure(orient+'.TScrollbar', troughcolor=panel, background=panel,
                        bordercolor=panel, lightcolor=panel, darkcolor=panel,
                        borderwidth=0, relief='flat', width=10, arrowsize=10, gripcount=0)
    style.layout('TNotebook', [])
    style.layout('TNotebook.Tab', [(prefix+'Notebook.tab', {'sticky': 'nswe', 'children': [
        ('Notebook.padding', {'sticky': 'nswe', 'children': [('Notebook.label', {'sticky': 'nswe'})]})]})])
    style.configure('TNotebook', background=panel, borderwidth=0, tabmargins=(0, 0, 0, 6))
    style.configure('TNotebook.Tab', padding=(12, 6), foreground=text, background=panel, font=('Segoe UI', 9))
    style.map('TNotebook.Tab', foreground=[('selected', text), ('active', text)])
    root.option_add('*TCombobox*Listbox.background', '#30313a' if mode == 'dark' else '#faf8fd')
    root.option_add('*TCombobox*Listbox.foreground', text)
    root.option_add('*TCombobox*Listbox.borderWidth', 0)
    root.option_add('*TCombobox*Listbox.highlightThickness', 0)
