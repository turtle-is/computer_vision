import face_recognition 
from face_recognition import face_locations, load_image_file
import face_recognition_models
from PIL import Image, ImageDraw


def face_rec():
   face_img = face_recognition.load_image_file("static/KMO_121188_03291_1_t218_214842.jpg")
   face_loc = face_recognition.face_locations(face_img)
   
   print(face_loc)
   print(f"Found {len(face_loc)} faces in the image")

   pil_image = Image.fromarray(face_img)
   draw1 = ImageDraw.Draw(pil_image)
   
   for(top, right, botton, left) in face_loc:
      draw1.rectangle(((left, top), (right, botton)), outline=(255, 255, 0), width=5)

   del draw1
   pil_image.save("static/output.jpg")   
   
   
   
def main():
   face_rec()
   
   
if __name__ == '__main__':
   main()