`timescale 1ns / 1ps

module variable_shift_reg #(parameter WIDTH = 8, parameter SIZE = 3) (
input clk,                                  //clock
input ce,                                   //clock enable                        
input rst,                                  //reset
input [WIDTH-1:0] d,                  //data in
output [WIDTH-1:0] out                 //data out
);
    
reg [WIDTH-1:0] sr [SIZE-1:0];            //the register that holds the data

generate
genvar i;
for(i = 0;i < SIZE;i = i + 1)
begin
    always@(posedge clk or posedge rst)
    begin
        if(rst)                          
        begin
            sr[i] <= 'd0;
        end
        else
        begin
            if(ce)                            //the shift register operates only when the clock-enable is active
            begin
                if(i == 'd0)
                begin
                    sr[i] <= d;
                end
                else
                begin
                    sr[i] <= sr[i-1];
                end
            end
        end
    end
end
endgenerate
assign out = sr[SIZE-1];
endmodule