select  count(*) from comments as c,  		posts as p,  		postLinks as pl,          postHistory as ph,          votes as v  where p.Id = c.PostId 	and p.Id = pl.PostId     and p.Id = ph.PostId     and p.Id = v.PostId  AND pl.LinkTypeId=1  AND p.ViewCount>=0  AND p.ViewCount<=7185  AND p.AnswerCount=1  AND p.CommentCount>=0  AND p.CommentCount<=22;