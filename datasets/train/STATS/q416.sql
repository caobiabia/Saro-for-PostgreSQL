select  count(*) from posts as p,  		postLinks as pl,          postHistory as ph where p.Id = pl.PostId 	and pl.PostId = ph.PostId  AND pl.LinkTypeId=1;
