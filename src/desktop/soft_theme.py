"""Rounded, scalable ttk surfaces, preserving native keyboard interactions."""
from PIL import Image, ImageDraw, ImageTk


def install(root, style):
    host = root._root()
    if not hasattr(host, '_soft_images'):
        host._soft_images = []
        def tile(fill, outline=None, radius=18, size=(80, 44), background='#f4f3f8'):
            scale = 3
            image = Image.new('RGBA', (size[0] * scale, size[1] * scale), background)
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle((1, 1, image.width - 2, image.height - 2), radius=radius * scale,
                                   fill=fill, outline=outline or fill, width=scale)
            image = image.resize(size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image, master=root)
            host._soft_images.append(photo)
            return photo
        def element(name, normal, *states, border=20):
            style.element_create(name, 'image', normal, *states, border=border, padding=0, sticky='nswe')
        element('Soft.card', tile('#ffffff', radius=18), border=20)
        for name, normal, hover, pressed, disabled, focus, background in (
            ('button', '#ffffff', '#f0eafa', '#e3d8f8', '#f4f1f8', '#b09ae9', '#f4f3f8'),
            ('cardbutton', '#ffffff', '#f0eafa', '#e3d8f8', '#f4f1f8', '#b09ae9', '#ffffff'),
            ('primary', '#7655d3', '#6847c4', '#5636ae', '#e7e0f3', '#b09ae9', '#f4f3f8'),
            ('cardprimary', '#7655d3', '#6847c4', '#5636ae', '#e7e0f3', '#b09ae9', '#ffffff'),
            ('danger', '#fff2f5', '#fbe4eb', '#f5d2dd', '#f7f1f4', '#ce90a2', '#ffffff')):
            element('Soft.' + name, tile(normal, '#e1dbe9', background=background),
                    ('disabled', tile(disabled, '#e7e1ee', background=background)),
                    ('pressed', tile(pressed, background=background)), ('active', tile(hover, background=background)),
                    ('focus', tile(normal, focus, background=background)))
        element('Soft.field', tile('#faf8fd', '#e0daeb', radius=12, background='#ffffff'),
                ('disabled', tile('#f4f1f7', '#e6e1ed', radius=12, background='#ffffff')),
                ('focus', tile('#ffffff', '#a78cde', radius=12, background='#ffffff')), border=14)
        arrow = Image.new('RGBA', (24, 20))
        ImageDraw.Draw(arrow).line([(7, 8), (12, 13), (17, 8)], fill='#827397', width=2)
        arrow = ImageTk.PhotoImage(arrow, master=root)
        host._soft_images.append(arrow)
        style.element_create('Soft.arrow', 'image', arrow, sticky='')
        element('Soft.trough', tile('#eee8f6', radius=6, size=(28, 12), background='#ffffff'), border=6)
        element('Soft.pbar', tile('#aa91e6', radius=6, size=(28, 12), background='#eee8f6'), border=6)
    style.layout('Rounded.TFrame', [('Soft.card', {'sticky': 'nswe'})])
    for name, element in (('TButton', 'button'), ('Primary.TButton', 'primary'), ('Danger.TButton', 'danger'), ('Card.TButton', 'cardbutton'), ('Card.Primary.TButton', 'cardprimary')):
        style.layout(name, [('Soft.' + element, {'sticky': 'nswe', 'children': [
            ('Button.padding', {'sticky': 'nswe', 'children': [('Button.label', {'sticky': 'nswe'})]})]})])
        style.configure(name, padding=(18, 10), borderwidth=0)
    style.configure('Photo.Card.TButton', padding=(10, 8))
    style.configure('Photo.Danger.TButton', padding=(10, 8))
    style.layout('TEntry', [('Soft.field', {'sticky': 'nswe', 'children': [
        ('Entry.padding', {'sticky': 'nswe', 'children': [('Entry.textarea', {'sticky': 'nswe'})]})]})])
    style.layout('TCombobox', [('Soft.field', {'sticky': 'nswe', 'children': [
        ('Soft.arrow', {'side': 'right', 'sticky': 'ns'}),
        ('Combobox.padding', {'sticky': 'nswe', 'children': [('Combobox.textarea', {'sticky': 'nswe'})]})]})])
    style.configure('TEntry', padding=(12, 9))
    style.configure('TCombobox', padding=(12, 9))
    style.layout('Horizontal.TProgressbar', [('Soft.trough', {'sticky': 'nswe', 'children': [
        ('Soft.pbar', {'side': 'left', 'sticky': 'ns'})]})])
    style.configure('Horizontal.TProgressbar', thickness=12, borderwidth=0)
    root.option_add('*TCombobox*Listbox.background', '#faf8fd')
    root.option_add('*TCombobox*Listbox.borderWidth', 0)
    root.option_add('*TCombobox*Listbox.highlightThickness', 0)
