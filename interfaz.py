from tkinter import *


class App:

	def __init__(self, master, t, d):
		self.frame = master
		self.tiempo = t
		self.distancia = d
		self.DrawLabel()

	def DrawLabel(self):
		velocidad = round(self.Speed(), 2)
		self.titulo = Label(self.frame, text="DATOS EXPERIMENTO FISICA I", font=("Arial", 16)).place(x=50, y=20)
		self.lbl_tiempo = Label(self.frame, text = "Tiempo del recorrido entre los puntos: ",font=("Arial", 10)).place(x=10, y=60)
		self.lbl_t = Label(self.frame, text=str(self.tiempo) + "seg", fg="blue",font=("Arial", 14)).place(x=250, y=60)
		self.lbl_distancia = Label(self.frame, text="Distancia entre los 2 puntos: ",font=("Arial", 10)).place(x=10, y=100)
		self.lbl_d2 = Label(self.frame, text=str(self.distancia) + "cm", fg="blue",font=("Arial", 14)).place(x = 250, y = 100)	

		#velocidad
		self.lbl_Velocidad = Label(self.frame, text="Velocidad en el recorrido: ", font=("Arial", 10)).place(x=10, y=150)
		self.lbl_v2 = Label(self.frame, text=str(velocidad) + "cm/s", fg="blue", font=("Arial", 14)).place(x=250, y=150)

	def Speed(self):
		#FORMULA :   V = D / T 
		return self.distancia / self.tiempo


'''
root = Tk()
root.geometry("400x400")
aplication = App(root, 5, 40)	

root.mainloop()	'''