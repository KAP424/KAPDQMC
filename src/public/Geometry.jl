# 120° basis
# PBC, OBC is not allowed
module Geometry

    export nn2idx,xy_i,i_xy,K_Matrix,area_index,nnidx_F

    function nnidx_F(Lattice,site)
        if Lattice=="SQUARE"
            Ns=prod(site)
            if length(site)==1
                nnidx=fill((0, 0), div(Ns,2), 2)
                count=1
                for i in 1:2:Ns
                    nn=nn2idx(Lattice,site,i)
                    for j in eachindex(nn)
                        nnidx[count,j]=(i,nn[j])
                    end
                    count+=1
                end
            elseif length(site)==2
                nnidx=fill((0, 0), div(Ns,2), 4)
                count=1
                for x in 1:site[1]
                    for y in 1:site[2]
                        if (x+y)%2==1
                            i=x+(y-1)*site[1]
                            nn=nn2idx(Lattice,site,i)
                            for j in eachindex(nn)
                                nnidx[count,j]=(i,nn[j])
                            end
                            count+=1
                        end
                    end
                end
            end
        elseif  occursin("HoneyComb", Lattice)
            Ns=prod(site)*2
            nnidx=fill((0, 0), div(Ns,2), 3)
            count=1
            for i in 1:2:Ns
                nn=nn2idx(Lattice,site,i)
                for j in eachindex(nn)
                    nnidx[count,j]=(i,nn[j])
                end
                count+=1
            end
        end
        return nnidx
    end

    function nn2idx(Lattice::String,site::Vector{Int64},idx::Int64)
        if Lattice=="SQUARE"
            if length(site)==2
                x,y=i_xy(Lattice,site,idx)
                nn=zeros(Int,4)
                nn[1]=xy_i(Lattice,site,mod1(x+1,site[1]),mod1(y,site[2]))
                nn[2]=xy_i(Lattice,site,mod1(x-1,site[1]),mod1(y,site[2]))
                nn[3]=xy_i(Lattice,site,mod1(x,site[1]),mod1(y+1,site[2]))
                nn[4]=xy_i(Lattice,site,mod1(x,site[1]),mod1(y-1,site[2]))
            else
                nn=[mod1(idx-1,site[1]),mod1(idx+1,site[1])]
            end

        elseif  Lattice=="HoneyComb120"
            nn=zeros(Int,3)
            x,y=i_xy(Lattice,site,idx)

            if mod(idx,2)==1
                nn[1]=idx+1
                nn[2]=xy_i(Lattice,site,mod1(x+1,site[1]),y)
                nn[3]=xy_i(Lattice,site,x,mod1(y-1,site[2]))

            else
                nn[1]=idx-1
                nn[2]=xy_i(Lattice,site,x,mod1(y+1,site[2]))-1
                nn[3]=xy_i(Lattice,site,mod1(x-1,site[1]), y)-1
            end

        elseif  Lattice=="HoneyComb60"
            nn=zeros(Int,3)
            x,y=i_xy(Lattice,site,idx)

            if mod(idx,2)==1
                nn[1]=idx+1
                nn[2]=xy_i(Lattice,site,mod1(x+1,site[1]),mod1(y-1,site[2]))
                nn[3]=xy_i(Lattice,site,x,mod1(y-1,site[2]))

            else
                nn[1]=idx-1
                nn[2]=xy_i(Lattice,site,x,mod1(y+1,site[2]))-1
                nn[3]=xy_i(Lattice,site,mod1(x-1,site[1]), mod1(y+1,site[2]))-1
            end
        else    
            error("Lattice: $(Lattice) is not allowed !")
        end
        return nn
    end

    function xy_i(Lattice::String,site::Vector{Int64},x::Int64,y::Int64)::Int64
        if Lattice=="SQUARE"
            if length(site)==2
                if x>site[1] || y>site[2]
                    println()
                    error("Error :($x,$y) Out of Lattice Range$(site)!")
                end
                return x+(y-1)*site[1]
            else
                error("Error : Dimension wrong")
            end


        elseif  occursin("HoneyComb", Lattice)
            if x>site[1] ||y>site[2]
                error("Error : Out of Lattice Range!")
            end
            return 2*(x+(y-1)*site[1])
        else
            error("Not support $Lattice")
        end
    end

    function i_xy(Lattice::String,site::Vector{Int64},i::Int64)
        if  Lattice=="SQUARE"
            if i>prod(site)
                error("Out of Lattice Range")
            else
                x=mod1(i,site[1])
                y=cld(i,site[1])
                return x::Int64,y::Int64
            end
        elseif  occursin("HoneyComb", Lattice)
            j=Int(ceil(i/2))
            return mod1(j,site[1]),Int(ceil(j/site[1]))
        else
            error("Not support $Lattice")
        end
    end

    function K_Matrix(Lattice::String,site::Vector{Int64})
        if Lattice=="SQUARE"
            Ns=prod(site)
            K=zeros(Float64,Ns,Ns)
            
        elseif occursin("HoneyComb", Lattice)
            Ns=prod(site)*2
            K=zeros(Float64,Ns,Ns)
        end

        for i in 1:Ns
            nnidx=nn2idx(Lattice,site,i)
            for idx in nnidx
                K[i,idx]=1
            end
        end
        return K
    end

    function area_index(Lattice::String,site::Vector{Int64},area::Tuple{Vector{Int64}, Vector{Int64}})::Vector{Int64}
        if Lattice=="SQUARE"
            if length(site)==1
                index=[x for x in area[1][1]:area[2][1]]
                return index

            elseif length(site)==2
                counter=1
                index=zeros(Int64,prod(area[2]-area[1]+[1,1]))
                for lx in area[1][1]:area[2][1]
                    for ly in area[1][2]:area[2][2]
                        index[counter]=xy_i(Lattice,site,lx,ly)
                        counter+=1
                    end
                end
                return index
            end
        elseif occursin("HoneyComb", Lattice)

            L=site[1]
            if area[1][1]==-1
                if Lattice=="HoneyComb60"
                    println("zigzag")
                    index=collect(4:2:xy_i(Lattice,site,L-1,1))
                    
                    for i in 2:div(2*L,3)
                        index=vcat(collect(xy_i(Lattice,site,2,i)-1   :1:  xy_i(Lattice,site,L-i,i) ),index)
                    end
                    return index        
                else
                    error("zigzag Only for HoneyComb60°")
                end
            
            elseif area[1][1]==-2
                if Lattice=="HoneyComb60"
                    index=Vector{Int64}()
                    println("beared")
                    for i in 2:div(2*L,3)
                        index=vcat(xy_i(Lattice,site,2,i)-1,index)
                        index=vcat(collect(xy_i(Lattice,site,3,i)-1   :1:  xy_i(Lattice,site,L-i+1,i)-1 ),index)
                    end
                    index=vcat(xy_i(Lattice,site,2,div(2*L,3)+1)-1,index)
                    return index 
                else
                    error("beared Only for HoneyComb60°")
                end
            else
                counter=1
                index=zeros(Int64,2*prod(area[2]-area[1]+[1,1]))
                for lx in area[1][1]:area[2][1]
                    for ly in area[1][2]:area[2][2]
                        index[counter]=xy_i(Lattice,site,lx,ly)-1
                        index[counter+1]=index[counter]+1
                        counter+=2
                    end
                end
                return index
            end
        end

    end

end
