from flexx import flx

class Example(flx.Widget):
	def init(self):
		with flx.HSplit():
			flx.Button(text='foo')
			with flx.VBox():
				flx.Widget(style='background:red',flex=1)
				flx.Widget(style='background:blue',flex=2)

app = flx.App(Example)
# app.export('example.html',link=0)

app.launch()
flx.run()
