select  count(*) from comments as c,  		posts as p,          postLinks as pl,          postHistory as ph,          votes as v,          badges as b  where p.Id = c.PostId     and p.Id = pl.RelatedPostId     and p.Id = ph.PostId     and p.Id = v.PostId 	and b.UserId = c.UserId  AND c.Score=0  AND pl.CreationDate<='2014-08-29 14:39:22'::timestamp  AND p.PostTypeId=1  AND p.ViewCount>=0  AND p.AnswerCount>=0  AND p.CommentCount=4  AND p.CreationDate>='2010-07-19 21:18:12'::timestamp  AND p.CreationDate<='2014-09-02 11:34:15'::timestamp  AND v.VoteTypeId=2  AND v.CreationDate>='2009-02-02 00:00:00'::timestamp;