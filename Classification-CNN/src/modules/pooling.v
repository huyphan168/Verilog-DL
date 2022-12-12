// Define the max pooling module
module max_pooling(input clk, input [15:0] data_in, output reg [15:0] data_out, parameter H = 256, parameter W = 256, parameter filters = 64);

    // Define the parameters for the max pooling operation
    parameter pool_width = 2;
    parameter pool_height = 2;

    // Define the storage registers for the max pooling operation
    reg [15:0] max_pooled_data [H/pool_width][W/pool_height][filters];

    // Loop through each pixel in the input image and perform the max pooling operation
    always @(posedge clk) begin
        for (int i = 0; i < H; i += pool_width) {
            for (int j = 0; j < W; j += pool_height) {
                for (int k = 0; k < filters; k++) {
                    max_pooled_data[i/pool_width][j/pool_height][k] = data_in[i][j][k];
                for (int l = 0; l < pool_width; l++) {
                    for (int m = 0; m < pool_height; m++) {
                            max_pooled_data[i/pool_width][j/pool_height][k] = max_pooled_data[i/pool_width][j/pool_height][k] > data_in[i+l][j+m][k] ? max_pooled_data[i/pool_width][j/pool_height][k] : data_in[i+l][j+m][k];
                    }
                }
            }
        }
    }
    end

    // Store the final max pooling output in the data_out register
    always @(posedge clk) begin
        data_out = max_pooled_data;
    end
endmodule