from app import StockMonitorApp
import customtkinter as ctk

def main():
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    app = StockMonitorApp()
    app.mainloop()

if __name__ == "__main__":
    main()
