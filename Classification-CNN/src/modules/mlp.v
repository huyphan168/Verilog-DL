// Define the fully connected layer module
module fully_connected_layer(input clk, input [7:0] data_in, input [7:0] weights [1024][64], input [7:0] biases [1024], output reg [7:0] data_out);

  // Define the storage registers for the fully connected operation
  reg [8:0] fc_result [1024];
  reg [8:0] fc_output [1024];

  // Loop through each input value and compute the fully connected operation
  for (int i = 0; i < 1024; i++) {
    for (int j = 0; j < 64; j++) {
      fc_result[i] += data_in[j] * weights[i][j];
    }
    fc_output[i] = fc_result[i] + biases[i];
  }

  // Store the final fully connected output in the data_out register
  data_out = fc_output;
endmodule
