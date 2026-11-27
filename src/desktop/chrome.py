"""Themed command menus with standard Windows caption controls preserved."""
import os
import tkinter as tk


PALETTES = {
    'light': dict(bg='#f4f3f8', fg='#242332', hover='#e8e3f1', popup='#ffffff',
                  border='#dedbe8', muted='#827b90', accent='#7655d3'),
    'dark': dict(bg='#1c1d23', fg='#eeeaf8', hover='#383143', popup='#26272e',
                 border='#42434e', muted='#91899e', accent='#b29af5'),
}


def titlebar(window, mode):
    """Use documented DWM attributes; older Windows keeps its native fallback.

    https://learn.microsoft.com/windows/win32/api/dwmapi/ne-dwmapi-dwmwindowattribute
    """
    if os.name != 'nt' or window.overrideredirect():
        return False
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.WinDLL('user32')
    user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    user32.GetAncestor.restype = wintypes.HWND
    hwnd = user32.GetAncestor(window.winfo_id(), 2)
    dwm = ctypes.WinDLL('dwmapi')
    dwm.DwmSetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
    dwm.DwmSetWindowAttribute.restype = ctypes.c_long
    def attribute(number, value):
        data = wintypes.DWORD(value)
        return dwm.DwmSetWindowAttribute(hwnd, number, ctypes.byref(data), ctypes.sizeof(data)) == 0
    def colorref(value):
        return int(value[1:3], 16) | int(value[3:5], 16) << 8 | int(value[5:7], 16) << 16
    palette = PALETTES[mode]
    supported = attribute(20, int(mode == 'dark'))
    attribute(35, colorref(palette['bg']))
    attribute(36, colorref(palette['fg']))
    attribute(34, colorref(palette['bg']))
    return supported


class MenuBar(tk.Frame):
    """Colorable menus backed by Tk command models and their live enable states.

    Alt+menu initial or F10 opens menus; arrows navigate and Enter activates.
    A grab dismisses outside clicks without leaking clicks to the photo workspace.
    """
    def __init__(self, root, sections):
        super().__init__(root, borderwidth=0, padx=12, pady=2)
        self._appearance_managed = True
        self.root, self.sections = root, sections
        self.mode = 'light'
        self.active = None
        self.popup = None
        self.rows = {}
        self.current = None
        self.previous_focus = None
        self.focus_check = None
        self.buttons = []
        self.pack(fill='x')
        for index, (label, _) in enumerate(sections):
            button = tk.Label(self, text=label, padx=10, pady=5, font=('Segoe UI', 9),
                              cursor='arrow', takefocus=True, borderwidth=0)
            button.pack(side='left')
            button.bind('<Button-1>', lambda _, i=index: self.toggle(i))
            button.bind('<Enter>', lambda _, i=index: self.hover(i))
            button.bind('<Leave>', lambda _: self.refresh())
            button.bind('<Return>', lambda _, i=index: self.open(i))
            button.bind('<space>', lambda _, i=index: self.open(i))
            button.bind('<Down>', lambda _, i=index: self.open(i))
            button.bind('<Right>', lambda _, i=index: self.focus_button((i+1) % len(sections)))
            button.bind('<Left>', lambda _, i=index: self.focus_button((i-1) % len(sections)))
            button.bind('<Escape>', lambda _: self.close())
            button.bind('<FocusIn>', lambda _: self.refresh())
            button.bind('<FocusOut>', lambda _: self.refresh())
            root.bind('<Alt-KeyPress-'+label[0].lower()+'>', lambda _, i=index: self.open(i), add='+')
            self.buttons.append(button)
        root.bind('<F10>', lambda _: self.open(0), add='+')
        root.bind('<Configure>', self.moved, add='+')
        self.bind('<Destroy>', self.destroyed, add='+')
        self.set_theme('light')

    def set_theme(self, mode):
        self.mode = mode
        self.palette = PALETTES[mode]
        self.configure(background=self.palette['bg'])
        self.refresh()

    def focus_button(self, index):
        self.buttons[index].focus_set()
        return 'break'

    def refresh(self):
        palette = self.palette
        for index, button in enumerate(self.buttons):
            focused = button.focus_get() == button
            button.configure(background=palette['hover'] if index == self.active or focused else palette['bg'], foreground=palette['fg'])
        if self.popup and self.popup.winfo_exists():
            self.popup.configure(background=palette['border'])
            self.popup_body.configure(background=palette['popup'])
            for index, row in self.rows.items():
                enabled = self.enabled(index)
                background = palette['hover'] if index == self.current and enabled else palette['popup']
                foreground = palette['fg'] if enabled else palette['muted']
                row.configure(background=background)
                for label in row.winfo_children():
                    label.configure(background=background, foreground=foreground)

    def enabled(self, index):
        model = self.sections[self.active][1]
        return str(model.entrycget(index, 'state')) != 'disabled'

    def hover(self, index):
        if self.popup:
            if index != self.active:
                self.open(index)
        else:
            self.buttons[index].configure(background=self.palette['hover'])

    def toggle(self, index):
        if self.popup and self.active == index:
            self.close()
        else:
            self.open(index)
        return 'break'

    def open(self, index):
        if not self.popup:
            self.previous_focus = self.root.focus_get()
        self.close(restore=False)
        self.active = index
        model = self.sections[index][1]
        popup = self.popup = tk.Toplevel(self.root)
        popup._appearance_managed = True
        popup.withdraw()
        popup.overrideredirect(True)
        popup.transient(self.root)
        self.popup_body = tk.Frame(popup, borderwidth=0, padx=4, pady=4)
        self.popup_body.pack(fill='both', expand=True, padx=1, pady=1)
        self.rows = {}
        last = model.index('end')
        for item in range(last+1 if last is not None else 0):
            if model.type(item) == 'separator':
                tk.Frame(self.popup_body, background=self.palette['border'], height=1).pack(fill='x', padx=8, pady=4)
                continue
            row = tk.Frame(self.popup_body, padx=10, pady=6)
            row.pack(fill='x')
            label = tk.Label(row, text=model.entrycget(item, 'label'), font=('Segoe UI', 9), anchor='w', borderwidth=0)
            label.pack(side='left', padx=(0, 30))
            accelerator = tk.Label(row, text=model.entrycget(item, 'accelerator'), font=('Segoe UI', 9), anchor='e', borderwidth=0)
            accelerator.pack(side='right')
            self.rows[item] = row
            for widget in (row, label, accelerator):
                widget.bind('<Enter>', lambda _, i=item: self.highlight(i))
                widget.bind('<ButtonRelease-1>', lambda _, i=item: self.invoke(i))
        self.current = next((i for i in self.rows if self.enabled(i)), None)
        self.refresh()
        popup.update_idletasks()
        button = self.buttons[index]
        x = min(button.winfo_rootx(), popup.winfo_screenwidth()-popup.winfo_reqwidth()-8)
        y = button.winfo_rooty()+button.winfo_height()+2
        if y+popup.winfo_reqheight() > popup.winfo_screenheight():
            y = max(0, button.winfo_rooty()-popup.winfo_reqheight())
        popup.geometry(f'+{max(0, x)}+{y}')
        popup.bind('<Escape>', lambda _: self.close())
        popup.bind('<Up>', lambda _: self.move(-1))
        popup.bind('<Down>', lambda _: self.move(1))
        popup.bind('<Home>', lambda _: self.edge(False))
        popup.bind('<End>', lambda _: self.edge(True))
        popup.bind('<Left>', lambda _: self.open((self.active-1) % len(self.sections)))
        popup.bind('<Right>', lambda _: self.open((self.active+1) % len(self.sections)))
        popup.bind('<Return>', lambda _: self.invoke(self.current))
        popup.bind('<space>', lambda _: self.invoke(self.current))
        popup.bind('<Tab>', lambda _: self.close())
        popup.bind('<ButtonPress-1>', self.outside)
        popup.bind('<Motion>', self.pointer)
        popup.bind('<FocusOut>', self.check_focus)
        popup.deiconify()
        popup.lift()
        popup.grab_set()
        popup.focus_set()
        return 'break'

    def highlight(self, index):
        self.current = index if self.enabled(index) else None
        self.refresh()

    def move(self, delta):
        enabled = [i for i in self.rows if self.enabled(i)]
        if enabled:
            position = enabled.index(self.current) if self.current in enabled else (-1 if delta > 0 else 0)
            self.current = enabled[(position+delta) % len(enabled)]
            self.refresh()
        return 'break'

    def edge(self, last):
        enabled = [i for i in self.rows if self.enabled(i)]
        if enabled:
            self.current = enabled[-1 if last else 0]
            self.refresh()
        return 'break'

    def invoke(self, index):
        if index is not None and self.enabled(index):
            model = self.sections[self.active][1]
            self.close()
            model.invoke(index)
        return 'break'

    def outside(self, event):
        x, y = event.x_root, event.y_root
        popup = self.popup
        if popup.winfo_rootx() <= x < popup.winfo_rootx()+popup.winfo_width() and popup.winfo_rooty() <= y < popup.winfo_rooty()+popup.winfo_height():
            return
        for index, button in enumerate(self.buttons):
            if button.winfo_rootx() <= x < button.winfo_rootx()+button.winfo_width() and button.winfo_rooty() <= y < button.winfo_rooty()+button.winfo_height():
                self.toggle(index)
                return 'break'
        self.close()
        return 'break'

    def pointer(self, event):
        # Under the popup grab, motion over the menu strip is sent here.
        for index, button in enumerate(self.buttons):
            if button.winfo_rootx() <= event.x_root < button.winfo_rootx()+button.winfo_width() and button.winfo_rooty() <= event.y_root < button.winfo_rooty()+button.winfo_height():
                if self.active != index:
                    self.open(index)
                break

    def check_focus(self, _):
        if self.focus_check:
            self.root.after_cancel(self.focus_check)
        self.focus_check = self.root.after_idle(self.lost_focus)

    def lost_focus(self):
        self.focus_check = None
        if self.popup and self.popup.winfo_exists() and self.popup.focus_displayof() != self.popup:
            self.close(restore=False)

    def moved(self, event):
        if event.widget == self.root and self.popup:
            self.close()

    def close(self, restore=True):
        if self.focus_check:
            self.root.after_cancel(self.focus_check)
            self.focus_check = None
        popup, self.popup = self.popup, None
        if popup and popup.winfo_exists():
            popup.grab_release()
            popup.destroy()
        self.rows = {}
        self.active, self.current = None, None
        if restore and self.previous_focus and self.previous_focus.winfo_exists():
            self.previous_focus.focus_set()
        if self.winfo_exists():
            self.refresh()
        return 'break'

    def destroyed(self, event):
        if event.widget == self:
            self.close(restore=False)
