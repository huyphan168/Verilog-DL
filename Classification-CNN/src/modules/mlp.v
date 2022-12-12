// Define the fully connected module
module fully_connected(input clk, input [15:0] data_in, output reg [15:0] data_out, parameter input_dim = 256, parameter output_dim = 256);

  // Define the number of DSP slices to use for the fully connected operation
  parameter num_dsp_slices = 8;

  // Define the parameters for the fully connected layer
  // (these will be specific to the trained CNN)
  parameter [15:0] weights [input_dim][output_dim];
  parameter [15:0] biases [output_dim];

  // Define the storage registers for the fully connected operation
  reg [15:0] fc_output [output_dim];

  // Loop through each DSP slice and perform the fully connected operation
  always @(posedge clk) begin
    for (int i = 0; i < output_dim; i++) {
      for (int j = 0; j < num_dsp_slices; j++) {
        DSP48E1 #(.INIT_N(0), .INIT_X(0)) dsp_slice(.A(data_in[j]), .B(weights[j][i]), .P(fc_output[i]));
      }
    fc_output[i] += biases[i];
  }
  end

  // Store the final fully connected output in the data_out register
  always @(posedge clk) begin
    data_out = fc_output;
  end
endmodule