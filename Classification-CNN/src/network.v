// Define the network module
module network(input clk, input [23:0] data_in, output reg [7:0] data_out);

  // Define the storage registers for the network
  reg [23:0] conv_output [128][128][64];
  reg [23:0] pool_output [64][64][64];
  reg [23:0] bn_output [64][64][64];
  reg [23:0] act_output [64][64][64];
  reg [23:0] fc_output [1024];
  reg [7:0] sm_output [2];

  // Define the convolutional layer
  convolutional_layer conv_layer(clk, data_in, conv_output);

  // Define the pooling layer
  pooling_layer pool_layer(clk, conv_output, pool_output);

  // Define the batch normalization layer
  batch_normalization_layer bn_layer(clk, pool_output, bn_output);

  // Load the weights, biases, and statistics for the batch normalization layer
  bn_layer.set_property(LOAD_WEIGHTS, "weights.bin");
  bn_layer.set_property(LOAD_BIASES, "biases.bin");
  bn_layer.set_property(LOAD_STATS, "stats.bin");

  // Define the activation layer
  activation_layer act_layer(clk, bn_output, act_output);

  // Define the fully connected layer
  fully_connected_layer fc_layer(clk, act_output, fc_output);

  // Load the weights and biases for the fully connected layer
  fc_layer.set_property(LOAD_WEIGHTS, "weights.bin");
  fc_layer.set_property(LOAD_BIASES, "biases.bin");

  // Define the softmax layer
  softmax_layer sm_layer(clk, fc_output, sm_output);

  // Store the final output of the network in the data_out register
  data_out = sm_output;
endmodule
