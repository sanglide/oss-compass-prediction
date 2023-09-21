from PIL import ImageFont,Image,ImageDraw

def add_font_in_pic(source_path,result_path,content):
    im1 = Image.open(source_path)
    draw = ImageDraw.Draw(im1)
    draw.text((100, 100), content, (0, 0, 0))
    im1.save(result_path)

if __name__=="__main__":
    im = Image.new("RGBA", (200, 200), 'white')
    image = im.convert("RGB")
    image.save("test.jpg")
    add_font_in_pic("test.jpg","test1.jpg","add some words")