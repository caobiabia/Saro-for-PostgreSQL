select  count(*) from comments as c,  		posts as p,  		postLinks as pl,          postHistory as ph,          votes as v  where p.Id = c.PostId 	and p.Id = pl.PostId     and p.Id = ph.PostId     and p.Id = v.PostId  AND c.CreationDate>='2010-08-03 15:26:43'::timestamp  AND c.CreationDate<='2014-09-03 08:06:25'::timestamp  AND ph.CreationDate>='2010-08-19 00:50:20'::timestamp  AND pl.LinkTypeId=1  AND p.ViewCount>=0  AND p.ViewCount<=10234  AND p.AnswerCount>=0  AND p.AnswerCount<=5  AND p.CommentCount>=0  AND p.CommentCount<=10;