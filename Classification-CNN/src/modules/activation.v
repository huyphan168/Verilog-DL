// Define the activation module
module activation(input clk, input [15:0] data_in, output reg [15:0] data_out);

  // Define the number of DSP slices to use for the activation operation
  int i;
  parameter num_dsp_slices = 8;
  parameter H = 256;
  parameter W = 256; 
  parameter num_filters = 64;

  // Define the storage registers for the activation operation
  reg [15:0] activated_data [0:H][0:W][0:num_filters];

  // Define the DSP slices for the activation operation
  reg [15:0] dsp_slice [0:num_dsp_slices];

  // Loop through each DSP slice and perform the activation
  for (i = 0; i < num_dsp_slices; i+=1) begin
    dsp_slice[i] = data_in[i] * weights[i] + biases[i];
    dsp_slice[i] = dsp_slice[i] > 0 ? dsp_slice[i] : 0;
    activated_data[i] = dsp_slice[i];
  end

  // Store the final activation output in the data_out register
  data_out = activated_data;
endmodule