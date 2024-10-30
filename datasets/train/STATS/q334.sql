select  count(*) from comments as c,  		posts as p,  		postLinks as pl,         votes as v   where p.Id = c.PostId 	and c.PostId = pl.PostId     and pl.PostId = v.PostId  AND v.VoteTypeId=2;
