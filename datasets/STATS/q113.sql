select  count(*) from comments as c,  		posts as p,  		postLinks as pl,          postHistory as ph,          votes as v  where p.Id = c.PostId 	and p.Id = pl.PostId     and p.Id = ph.PostId     and p.Id = v.PostId  AND ph.CreationDate>='2010-08-19 13:37:35'::timestamp  AND ph.CreationDate<='2014-08-17 07:54:57'::timestamp  AND p.PostTypeId=2  AND p.ViewCount>=0  AND p.AnswerCount>=0  AND p.AnswerCount<=7  AND v.VoteTypeId=2  AND v.CreationDate>='2010-07-19 00:00:00'::timestamp;