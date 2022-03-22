# Load a ONNX model
import numpy as np
import torch

from helpers import trt_helper
from helpers import trt_int8_calibration_helper as int8_helper
import time
from calib import ImageBatchStream
import argparse


class CNN(torch.nn.Module):
    def __init__(self, num_classes=10,):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Conv2d(3,16,3)
        self.layer2 = torch.nn.Conv2d(16,64,5)
        self.relu = torch.nn.ReLU()
        
        # TAKE CARE HERE
        # Ceil_mode must be False, because onnx eporter does NOT support ceil_mode=True
        self.max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, ceil_mode=False) 
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1,1)) 
        
        self.fc = torch.nn.Linear(64,num_classes)
        self.batch_size_onnx = 0
        # FLAG for output ONNX model
        self.export_to_onnx_mode = False                  
      
    def forward_default(self, X_in):
        print("Function forward_default called! \n")
        x = self.layer1(X_in)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        
        # Such an operationt is not deterministic since it would depend on the input and therefore would result in errors
        length_of_fc_layer = x.size(1) 
        x = x.view(-1, length_of_fc_layer)
        
        x = self.fc(x)
        return x

    def forward_onnx(self, X_in):
        print("Function forward_onnx called! \n")
        x = self.layer1(X_in)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        assert self.batch_size_onnx > 0
        length_of_fc_layer = 64 # For exporting an onnx model that fit the TensorRT, processes here should be DETERMINISITC!
        x = x.view(self.batch_size_onnx, length_of_fc_layer) # 
        x = self.fc(x)
        return x

    def __call__(self, *args,**kargs):
        if self.export_to_onnx_mode:
            return self.forward_onnx(*args,**kargs)
        else:
            return self.forward_default(*args,**kargs)

def generate_onnx_model(onnx_model_path, img_size, batch_size):
    #model = CNN(10)
    model_path = '/home/jiuling/data/demo1/backbone_with_64dim_0816.onnx'
    model = torch.load(model_path, map_location=torch.device('cpu'))
   
    # This is for ONNX exporter to track all the operations inside the model
    batch_size_of_dummy_input = batch_size # Any size you want
    dummy_input = torch.zeros((batch_size_of_dummy_input,)+img_size, dtype=torch.float32)

    model.batch_size_onnx = batch_size_of_dummy_input
    model.export_to_onnx_mode = True
    input_names = [ "data" ]
    #output_names = [ "644","650"] # Multiple inputs and outputs are supported
    output_names = ['1879']
    with torch.no_grad():
        # If verbose is set to False. The information below won't displayed
        torch.onnx.export(model, dummy_input, onnx_model_path, verbose=True, input_names=input_names, output_names=output_names)

def main():
    # Prepare a dataset for Calibration
    img_size = (3,116,100)
    #onnx_model_path = 'model_128.onnx'
    #onnx_model_path = 'backbone_with_64dim_0816.onnx'
    #onnx_model_path = '/home/jiuling/face/lzx_train/model_cspnet.onnx_sim_model.onnx'
    #onnx_model_path = '/home/jiuling/face/lzx_train/model_res2net.onnx_sim_model.onnx'
    #onnx_model_path = '/data0/tools/PyTorch_ONNX_TensorRT/models/backbone_1012_no64dim.onnx'
    #onnx_model_path = '/data0/tools/PyTorch_ONNX_TensorRT/models/backbone_1014_tmp.onnx'
    #onnx_model_path = '/data0/tools/PyTorch_ONNX_TensorRT/models/backbone_1015_tmp3.onnx'
    #onnx_model_path = '/data0/tools/PyTorch_ONNX_TensorRT/models/backbone_with_64dim_0816.onnx'
    #onnx_model_path = '/data0/tools/PyTorch_ONNX_TensorRT/models/pfld.onnx'
    #generate_onnx_model(onnx_model_path, img_size, batch_size)
    onnx_model_path = './models/backbone_1111.onnx'
    save_suffix = 'small_calib_png'

    parser = argparse.ArgumentParser(description='calib information')
    parser.add_argument('--calib_list', dest='calib_list', type=str, help='',default='small_calib_png.lst')
    parser.add_argument('--batch_size', dest='batch_size', type=int, help='',default=512)
    parser.add_argument('--input_channel', dest='input_channel', type=int, help='',default=3)
    parser.add_argument('--input_width', dest='input_width', type=int, help='',default=100)
    parser.add_argument('--input_height', dest='input_height', type=int, help='',default=116)
    parser.add_argument('--batch_cache_dir',dest='batch_cache_dir',type=str,help='',default='./batch_caches/')
    args = parser.parse_args()
    dev_id = 1

    dataset = np.random.rand(1000,*img_size).astype(np.float32)
    max_batch_for_calibartion = 256
    transform = None
    batch_size = args.batch_size

    # Prepare a stream
    #calibration_stream = int8_helper.ImageBatchStreamDemo(dataset, transform, max_batch_for_calibartion, img_size)
    calibration_stream = ImageBatchStream(args)
    #batch_size = 1000

    engine_model_path = onnx_model_path+".%s-int8.trt" % save_suffix
    calib_file = onnx_model_path + ".%s_calib.bin" % save_suffix
    engine_int8 = trt_helper.get_engine(batch_size,onnx_model_path,engine_model_path, fp16_mode=False, int8_mode=True, calibration_stream=calibration_stream, save_engine=True,calibration_cache=calib_file,dev_id=dev_id)
    assert engine_int8, 'Broken engine'
    context_int8 = engine_int8.create_execution_context() 
    inputs_int8, outputs_int8, bindings_int8, stream_int8 = trt_helper.allocate_buffers(engine_int8)

    engine_model_path = onnx_model_path+".%s-fp16.trt" % save_suffix
    engine_fp16 = trt_helper.get_engine(batch_size,onnx_model_path,engine_model_path, fp16_mode=True, int8_mode=False, save_engine=True)
    assert engine_fp16, 'Broken engine'
    context_fp16 = engine_fp16.create_execution_context() 
    inputs_fp16, outputs_fp16, bindings_fp16, stream_fp16 = trt_helper.allocate_buffers(engine_fp16)

    engine_model_path = onnx_model_path+".%s-fp32.trt" % save_suffix
    engine = trt_helper.get_engine(batch_size,onnx_model_path,engine_model_path, fp16_mode=False, int8_mode=False, save_engine=True)
    assert engine, 'Broken engine'
    context = engine.create_execution_context() 
    inputs, outputs, bindings, stream = trt_helper.allocate_buffers(engine)

    total_time_int8 = []
    total_time_fp16 = []
    total_time = []
    for i in range(1, dataset.shape[0]):
        x_input = dataset[i]
        inputs_int8[0].host = x_input.reshape(-1)

        tic_int8 = time.time()
        trt_helper.do_inference(context_int8, bindings=bindings_int8, inputs=inputs_int8, outputs=outputs_int8, stream=stream_int8)
        toc_int8 = time.time()
        total_time_int8.append(toc_int8 -tic_int8 )

        tic_fp16 = time.time()
        trt_helper.do_inference(context_fp16, bindings=bindings_fp16, inputs=inputs_fp16, outputs=outputs_fp16, stream=stream_fp16)
        toc_fp16 = time.time()
        total_time_fp16.append(toc_fp16 -tic_fp16 )

        tic = time.time()
        trt_helper.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        toc = time.time()
        total_time.append(toc -tic)
    
    print('Toal time used by engine_int8: {}'.format(np.mean(total_time_int8)))
    print('Toal time used by engine_fp16: {}'.format(np.mean(total_time_fp16)))
    print('Toal time used by engine: {}'.format(np.mean(total_time)))


if __name__ == '__main__':
    main()
