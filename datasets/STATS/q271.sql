select  count(*) from comments as c,  		posts as p,          postHistory as ph,          votes as v,          badges as b,          users as u  where u.Id = p.OwnerUserId     and u.Id = b.UserId     and p.Id = c.PostId     and p.Id = ph.PostId     and p.Id = v.PostId  AND c.CreationDate>='2010-07-20 17:06:12'::timestamp  AND c.CreationDate<='2014-09-10 03:05:33'::timestamp  AND ph.PostHistoryTypeId=3  AND p.Score<=16  AND p.ViewCount<=6191  AND p.AnswerCount=2  AND u.CreationDate>='2011-04-28 22:38:17'::timestamp  AND u.CreationDate<='2014-09-04 19:40:22'::timestamp  AND v.VoteTypeId=2  AND v.CreationDate<='2014-09-08 00:00:00'::timestamp;