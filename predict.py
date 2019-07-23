import eplon_voice_generate
import eplon_voice_predict

eplon_voice_generate.eplon_voice_generate_predict_data()
key = input(" PUSH [c] naive cnn [s] sercret cnn :")
if(key=='c'):
    netname = 'net1'
else:
    netname = 'net2'
eplon_voice_predict.eplon_voice_predict(netname)
input("- PUSH ENTER KEY TO EXIT -")
