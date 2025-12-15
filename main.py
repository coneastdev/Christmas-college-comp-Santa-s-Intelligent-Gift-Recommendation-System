import customtkinter

class Main(customtkinter.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Santa's Intelligent Gift Recommendation System")
        self.geometry(f"{500}x{500}")

        

if __name__ == "__main__":
    main = Main()
    main.mainloop()