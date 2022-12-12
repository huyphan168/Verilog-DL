module mac_manual #(
	parameter N = 16,
    parameter Q = 12
    )(
    input clk,sclr,ce,
    input [N-1:0] a,
    input [N-1:0] b,
    input [N-1:0] c,
    output reg [N-1:0] p
    );

always@(posedge clk,posedge sclr)
 begin
    if(sclr)
    begin
        p<=0;
    end
    else if(ce)
    begin
        p <= (a*b+c);                   //performs the multiply accumulate operation
    end
 end
endmodule