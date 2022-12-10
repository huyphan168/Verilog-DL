// Define the activation module
module activation(input clk, input [7:0] data_in, output reg [7:0] data_out);

  // Define the storage registers for the activation operation
  reg [7:0] activated_data [256][256][64];

  // Loop through each pixel in the input image and perform the activation
  for (int i = 0; i < 256; i++) {
    for (int j = 0; j < 256; j++) {
      for (int k = 0; k < 64; k++) {
        // Perform the ReLU activation function on the input data
        activated_data[i][j][k] = data_in[i][j][k] > 0 ? data_in[i][j][k] : 0;
      }
    }
  }

  // Store the final activation output in the data_out register
  data_out = activated_data;
endmodule
