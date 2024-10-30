select  count(*) from comments as c,  		postLinks as pl,  		postHistory as ph,          votes as v where pl.PostId = c.PostId 	and c.PostId = ph.PostId 	and ph.PostId = v.PostId  AND v.VoteTypeId=2;
